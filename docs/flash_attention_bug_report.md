# NVIDIA Flash Attention Prefill 输出 NaN 问题报告

> 记录 `test/test_infer.py --device nvidia` 从跑不通到跑通期间定位并修复的 GPU 推理 NaN 问题。本报告可作为后续排查数值问题的参考。

## 0. 背景

任务目标：打通 NVIDIA GPU 上的模型推理测试 `python test/test_infer.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --device nvidia`，且 CPU 测试不受影响。模型为 DeepSeek-R1-Distill-Qwen-1.5B（nlayer=28、hs=1536、nh=12、nkvh=2、dh=128、di=8960、vocab=151936，bf16），运行环境 RTX 4090 + CUDA 12.8。

`self_attention` 在 NVIDIA 上的分流逻辑（`src/ops/self_attention/op.cpp`）：

```cpp
if (seqlen > 1 && total_len == seqlen && d == 128 && dv == 128) {
    nvidia::flash_attention(...);               // prefill：K/V 分块 + online softmax，TILE_Q=8
} else if (seqlen == 1 && d == 128 && dv == 128 && total_len > kSplitKVThreshold) {
    nvidia::flash_attention_decode_splitkv(...); // decode：长 KV，split-kv
} else if (seqlen == 1 && d == 128 && dv == 128) {
    nvidia::flash_attention_decode(...);        // decode：短 KV
} else {
    nvidia::self_attention(...);                // V1 兜底
}
```

## 1. 问题分析

### 1.1 现象

- `test_infer.py --device nvidia --test` 断言失败：llaisys 生成 token 与 HF 参考不一致；
- 逐层 dump 显示 `last_logits` **全为 NaN**，argmax 返回垃圾 token id（如 `139707031638018`）；
- `diag_ops.py`（真实模型 shape 的算子对拍）首次运行 argmax 失败（llaisys=32946 vs torch=274），随后 3 轮全部通过——初看像"随机竞态"；
- 但 `test_infer.py` 复现时 **每次运行 L0 层 attn_val 必为 NaN**——又是确定性的。

"瞬态"与"必现"两个互相矛盾的表象，最终被证明是同一个根因（见 1.3）。

### 1.2 NaN 传播链

下游看到的两个症状都只是 NaN 的"下游效应"：

1. **logits 全 NaN**：NaN 从 L0 层 self_attention 输出（attn_val）进入残差流，逐层传播到最后的 lm_head；
2. **argmax 返回垃圾 id**：argmax kernel 的 `best_idx` 初始化为 0，而所有比较 `x > best` 对 NaN 均为 false，`best_idx` 始终未被更新——返回的是**未初始化的索引值**，数值大小随内存内容变化，因此同一 bug 时而表现为 32946、时而表现为 139707031638018，产生了"瞬态"的假象。

### 1.3 根因机制

`flash_attention_prefill_nvidia.cu` 的 chunk 循环里，K/V 块经 shared memory 协作搬运：

```cuda
for (size_t flat = threadIdx.x; flat < Bc * d; flat += blockDim.x) {
    size_t dim = flat % d;
    size_t row_in_chunk = flat / d;
    if (j + row_in_chunk <= tile_max_limit) {   // 只有 <= tile_max_limit 的行才被写入
        K_chunk[flat] = k[...];
        V_chunk[flat] = v[...];
    }
}
```

当 `seqlen` 不是 `Bc` 的整数倍时，chunk 尾部的行**不会被写入**——这些 shared memory 槽位残留着上一个 kernel（前序 cuBLAS GEMM 等）留下的位型。本轮排查中前序 kernel 固定遗留 `±inf` 位型（因此每次运行必现 NaN）。

后续 softmax 概率经 causal mask 处理：被掩码位置 `p = 0`。累加时：

```
acc += p * V_chunk[残留行]   →   0 × inf = NaN   （IEEE 754）
```

即使残留位型碰巧是有限值，`0 × 有限值 = 0` 也恰好无害——这正是"位型运气"：看似随机，其实只取决于上一个 kernel 留下了什么。

**根因一句话**：分块 kernel 部分填充 shared memory，被 causal mask 置 0 的概率与未填充区域（残留 inf）相乘，`0 × inf = NaN`。

## 2. 问题定位

定位过程逐层收缩，每步都有可复现的证据：

| 步骤 | 手段 | 结论 |
| --- | --- | --- |
| 1 | `test_infer.py --device nvidia` 复现 | `last_logits` 全 NaN，输出与 HF 不一致 |
| 2 | `diag_ops.py`：真实模型 shape 对拍 8 算子 | 仅 argmax 首次失败（32946 vs 274），其余算子全过；后 3 轮 argmax 又全过 |
| 3 | `diag_argmax.py`：argmax 隔离测试（多 numel/dtype、已知最大位置） | 对**有限**输入全部正确 → argmax kernel 本身没问题，问题在其输入 |
| 4 | `test_infer.py` 加逐层 dump | NaN **首次出现**在 L0 层 attn_val（self_attention 输出）；该层输入 q/k/v 全部有限 → NaN 由 self_attention 内部产生 |
| 5 | dump cached_k/cached_v | 与 rope 后的 k、attn 前的 v 逐元素一致 → KV cache 追加（D2D memcpy 修复）正确，排除 |
| 6 | 静态分析 kernel 的填充边界 | chunk 尾部 `j + row_in_chunk <= tile_max_limit` 之外的行不写入 → 残留位型 + `0 × inf = NaN` |

关键证据链：**输入有限 → 输出 NaN → 只可能是 kernel 内部产生了 inf/NaN 参与运算 → 唯一的可疑源是未初始化的 shared memory 行**。

## 3. 解决方法

### 3.1 核心修复

`src/ops/self_attention/nvidia/flash_attention_prefill_nvidia.cu`：每轮 chunk 搬运前先把 K/V 整块清零，再同步，然后做条件填充：

```cuda
for (int j = 0; j <= tile_max_limit; j += Bc) {
    // 先把整块清零：chunk 尾部超出 tile_max_limit 的行不会被写入，
    // 若残留上一位使用者的 inf/NaN 位型，被掩码行 p=0 相乘时 0*inf 会得到 NaN。
    for (size_t flat = threadIdx.x; flat < Bc * d; flat += blockDim.x) {
        K_chunk[flat] = 0.0f;
        V_chunk[flat] = 0.0f;
    }
    __syncthreads();
    for (size_t flat = threadIdx.x; flat < Bc * d; flat += blockDim.x) {
        size_t dim = flat % d;
        size_t row_in_chunk = flat / d;
        if (j + row_in_chunk <= tile_max_limit) {
            K_chunk[flat] = k[(j + row_in_chunk) * nkvhead * d + kvh * d + dim];
            V_chunk[flat] = v[(j + row_in_chunk) * nkvhead * dv + kvh * dv + dim];
        }
    }
    __syncthreads();
    // ... 后续不变
}
```

清零后未填充行位型为 0，`0 × 0 = 0`，从机制上消除了 NaN 的可能。

### 3.2 排查过程中顺带修复的问题

| 文件 | 问题 | 修复 |
| --- | --- | --- |
| `src/models/kv_cache.cpp` | KV cache 追加走 `load()`（H2D 路径），device 间拷贝语义错误 | 改用 `memcpy_sync(..., LLAISYS_MEMCPY_D2D)`；CPU 下退化为普通 memcpy，不受影响 |
| `src/models/qwen2/model.cpp` | argmax 结果直接解引用 device 指针（CPU 下碰巧正确） | 增加 `fetch_scalar_int64_`，device 张量经 D2H 读回；`make_pos_ids_` 改为 host 构造再 load |

### 3.3 构建与验证

```bash
export PATH=$PATH:/root/.local/bin    # xmake 安装位置
export XMAKE_ROOT=y                   # root 容器运行 xmake 需要
xmake build && xmake install -o .     # after_install 将 .so 部署到 python/llaisys/libllaisys/
```

验证结果：

| 测试 | 结果 |
| --- | --- |
| `test_infer.py --model <模型> --device nvidia --test`（硬断言） | ✅ Test passed!，25 token 与 HF 完全一致 |
| `test_infer.py --device nvidia`（默认 128 steps） | ✅ 91 token 与 HF 逐 token 一致 |
| `test_infer.py --test --device cpu` | ✅ 通过（CPU 后端不受影响） |
| `test_ops.py --device nvidia` 连续 3 轮 | ✅ 全部通过（argmax 的"瞬态失败"不再出现） |

修复后 GPU 推理打通，顺带获得性能优势：同 128 steps 下 llaisys GPU 0.61s vs torch GPU 2.13s（详见 REPORT.md 1.4）。

### 3.4 保留的排查工具

排查用的临时脚本已整合为两个可复用诊断脚本（`test/diag_ops.py`、`test/diag_infer.py`）：

- `diag_ops.py`：算子级对拍。真实模型 shape 的 8 算子对拍 + argmax 隔离 + self_attention 三分流（prefill/decode/splitkv）+ H2D/D2H memcpy 自检。推理输出不对时先跑它，一步定位"哪个算子坏了"。
- `diag_infer.py`：推理级对拍。端到端 HF vs llaisys 生成对比 + `--replay` 模式用真实权重逐层回放 prefill（每步对拍），NaN 类问题用它做逐层 bisect。

## 4. 经验总结

1. **分块 kernel 部分填充 shared memory 必须先清零**：flash attention 中"部分填充 smem + 掩码概率乘以未填充区域"是 NaN 的经典来源。不能依赖"位型运气"（残留有限值就恰好无害），必须清零后再条件填充，从机制上消灭 `0 × inf`。
2. **NaN 定位用逐层 bisect，在"首次出现"的算子边界停下**：NaN 会沿残差流传播，在 logits/argmax 末端观察只能看到垃圾 token；从 L0 逐层对比，发现"输入有限、输出 NaN"就锁定了唯一可疑源。
3. **`0 × inf = NaN` 是 IEEE 754 的确定性行为**："每次都必现"的 NaN 不是竞态，恰恰说明残留位型是固定的（前序 kernel 确定性地遗留 inf）——"瞬态"的表象可能只是下游 argmax 未初始化索引的随机值，不要被它误导去追竞态。
4. **argmax 返回垃圾 id 是 NaN 的下游症状**：NaN 参与比较全部为 false，`best_idx` 从不更新。先修上游 NaN，不要在下游 kernel 补保护——那只是掩盖症状。
5. **三层用法**：逐字节 diff 排除实现差异 → 读其调试文档排除已知 bug → 对照实现结构确认边界条件。排除了所有"已知正确"的部分，剩下的未初始化 smem 就是唯一候选。
6. **环境坑备忘**：xmake 需 `PATH` + `XMAKE_ROOT=y`；Python 包须可编辑安装（`pip install -e`），否则测试加载旧 `.so`；device 张量必须 D2H 读回，不能直接解引用；后台链式命令会因 cwd 漂移静默失败，拆分并显式 `cd`。
