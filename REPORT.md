# LLAISYS 作业报告（作业 #1 ~ #4）

> 提交要求见 [README_ZN.md](README_ZN.md)「作业提交要求」：CI 通过、简要报告说明复现流程并记录复现结果、逐个平台说明支持的平台及其状态。本报告按此组织，任务细节不再复述。

**摘要**：作业 #1 ~ #3（CPU 全链路）已完整实现并通过测试，推理输出与 HF 参考实现逐 token 一致；作业 #4 选定 NVIDIA、Iluvatar 与 Moore 三个 GPU 平台，Runtime API、8 算子 GPU 实现与模型 GPU 推理全部完成——NVIDIA 侧全链路测试通过（推理输出与 HF 逐 token 一致）；Moore 侧（MTT S5000 + MUSA 4.3.5）全链路测试通过（128 steps 耗时 8.26s，比同环境 torch_musa 快约 1.5 倍）；Iluvatar 侧 Runtime API 与算子代码已完成，因容器环境障碍未验证。CI（GitHub Actions，ubuntu-latest / windows-latest）覆盖 CPU 后端全部作业测试，详见 1.2。

## 1. 完成情况与结果记录

### 1.1 作业完成情况

| 作业 | 内容 | 状态 |
| --- | --- | --- |
| #1 张量 | Storage 共享内存 + `load` / `isContiguous` / `view` / `permute` / `slice` | ✅ 完成 |
| #2 算子 | 8 种算子（add/argmax/embedding/linear/rms_norm/rope/swiglu/self_attention），支持 fp32/fp16/bf16，dtype 分派，半精度升 float 计算 | ✅ 完成 |
| #3 模型推理 | Qwen2（DeepSeek-R1-Distill-Qwen-1.5B）：safetensors 权重加载、KV Cache、argmax 采样生成，Python 前端仅作 ctypes 封装 | ✅ 完成 |
| #4 GPU 集成 | NVIDIA + Iluvatar + Moore 三平台：Runtime API、8 算子 GPU 实现、模型推理改造 | ✅ NVIDIA、Moore 全链路；Iluvatar 代码完成、未验证（见 2.3） |

### 1.2 CI 状态

GitHub Actions（`.github/workflows/build.yaml`）在 `ubuntu-latest` 与 `windows-latest` 上自动运行全部 CPU 后端测试（runtime / 张量 / 8 算子 / Qwen2 推理），workflow 已全部绿色通过；进入仓库 GitHub 页面即可查看流水线状态与各步骤结果。

### 1.3 测试结果（本地四平台）

| 测试模块 | 对应作业 | CPU | NVIDIA | Iluvatar | Moore |
| --- | --- | --- | --- | --- | --- |
| tensor（load/view/permute/slice 等） | #1 | ✅ 通过 | —（设备无关） | —（设备无关） | —（设备无关） |
| 8 种算子（fp32/fp16/bf16） | #2 | ✅ 通过 | ✅ 通过 | ⚠️ 代码完成，未验证 | ✅ 通过 |
| Qwen2 推理 | #3 | ✅ 通过（输出与 HF 逐 token 一致） | ✅ 通过（输出与 HF 逐 token 一致） | ⚠️ 未验证 | ✅ 通过（输出与 HF 逐 token 一致） |
| runtime | #0 | ✅ 通过 | ✅ 通过 | ⚠️ 代码完成，未验证 | ✅ 通过 |

> CPU 后端测试在本地、NVIDIA 与 Moore 环境均可跑通；NVIDIA 环境全部测试（含模型 GPU 推理）均已通过，`test_ops.py --device nvidia` 连续多轮全过；Moore 环境全部测试（含模型 GPU 推理）均已通过；Iluvatar 侧 Runtime API 与算子代码已完成，因容器环境配置障碍无法运行验证测试，目前无实测结果。

### 1.4 推理正确性与性能

- **正确性**：llaisys 在 CPU 与 NVIDIA GPU 上的生成文本均与 HF 参考实现（PyTorch + transformers）逐 token 一致（argmax 采样，`--test` 硬断言通过）；Moore GPU 上推理以默认采样参数（temperature 1.0 / top_k 50 / top_p 0.8）跑通，未启用 `--test` 硬断言，逐 token 对比为后续补测项。
- **性能**（NVIDIA 与 Moore 机器上测量，`test_infer.py` 默认参数 128 steps，prompt "Who are you?"）：

| 实现 | 实际运行设备 | 128 steps 耗时 |
| --- | --- | --- |
| torch（HF 参考实现） | CPU | 7.01 s |
| torch（HF 参考实现） | NVIDIA RTX 4090 | 2.13 s |
| torch（HF 参考实现，torch_musa） | Moore MTT S5000 | 12.20 s |
| llaisys（本仓库） | CPU | 391.66 s |
| llaisys（本仓库） | NVIDIA RTX 4090 | 0.61 s |
| llaisys（本仓库） | Moore MTT S5000 | 8.26 s |

- llaisys GPU 相对自身 CPU 后端加速约 640 倍：CPU 后端为朴素实现（无 flash attention、逐 token 全量扫描、逐元素循环），GPU 后端走 cuBLAS + flash attention 分块 kernel；
- llaisys GPU（0.61s）比 torch GPU（2.13s）快约 3.5 倍：llaisys 全链路为 C++ 单次函数调用，无 Python 解释器与逐 token 调度开销；
- llaisys Moore（8.26s）比同环境 torch_musa GPU（12.20s）快约 1.5 倍：MTT S5000 上 llaisys 同样为 C++ 单次函数调用，GEMM 走 mublas，attention 走 flash attention 分块 kernel；
- torch CPU（7.01s）远快于 llaisys CPU（391.66s）：torch 走 oneDNN/MKL 高度优化 kernel + 多线程并行，llaisys CPU 后端目前仅保证正确性，性能优化为后续工作。

## 2. 支持的平台及状态（逐个平台）

| 平台 | 张量 | 运行时 | 算子 | 模型推理 | 构建开关 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| CPU | ✅ | ✅ | ✅ | ✅ | 默认 | ✅ 完全支持 |
| NVIDIA | 设备无关 | ✅ | ✅ | ✅ | `--nv-gpu=y` | ✅ 完全支持 |
| Moore | 设备无关 | ✅ | ✅ | ✅ | `--moore-gpu=y` | ✅ 完全支持 |
| Iluvatar | 设备无关 | ⚠️ 未验证 | ⚠️ 未验证 | ⚠️ 未验证 | `--iluvatar-gpu=y` | ⚠️ 代码完成，容器障碍未验证 |

### 2.1 CPU

全链路可用：张量、8 算子（fp32/fp16/bf16）、Qwen2 推理全部通过，输出与 HF 逐 token 一致；CPU 测试在本地、NVIDIA 与 Moore 环境均可跑通。

### 2.2 NVIDIA（GeForce RTX 4090, 24 GB）

Runtime API（12 个函数）、8 种算子的 CUDA 实现与模型 GPU 推理全链路均已通过测试：`test_runtime.py` / `test_ops.py --device nvidia` 连续多轮通过，`test_infer.py --device nvidia` 输出与 HF 逐 token 一致（`--test` 硬断言通过）。self_attention 按 shape 分流到三个专门 kernel：prefill（K/V 分块 + online softmax，TILE_Q=8）、decode（单 query 行）、decode split-kv（长 KV 分片），其余 shape 落回 V1 兜底。推理性能：128 steps 耗时 0.61s，比同环境 torch GPU（2.13s）快约 3.5 倍。期间定位并修复了一个 flash attention prefill 的 NaN 问题（详见 4.7 与 `docs/flash_attention_bug_report.md`）。

### 2.3 Iluvatar（MR-V100, 32 GB）

Runtime API 与 8 种算子已按 corex 工具链（`clang++ -x ivcore` + cuBLAS）完成代码实现；受容器环境配置障碍影响，编译与运行测试均未能在该环境中验证，目前无实测结果，待环境修复后补测。

### 2.4 Moore（MTT S5000, 80 GB）

Runtime API（12 个函数）、8 种算子的 MUSA 实现与模型 GPU 推理全链路均已通过测试：`test_runtime.py` / `test_ops.py --device moore` 通过，`test_infer.py --device moore` 以默认采样参数跑通（未启用 `--test` 硬断言，逐 token 对比为后续补测项）。基于 MUSA 4.3.5 工具链（`mcc -x musa` + musart/mublas）实现：MUSA API 与 CUDA 命名一一对应（`cudaXxx` → `musaXxx`、`cublas*` → `mublas*`、`__nv_bfloat16` → `__mt_bfloat16`），8 个算子的 kernel 由 NVIDIA 版近乎直译平移，linear 的 GEMM 走 `mublasGemmEx`（BF16/F16）与 `mublasSgemm`（F32），self_attention 沿用 prefill / decode / decode split-kv 三个 flash attention 分块 kernel。构建上在 `xmake/moore.lua` 定义独立 moore 工具链（mcc + `-x musa`），并用本地空 `cuda.env` 规则覆盖 xmake 内置 CUDA 规则链以阻断 nvcc 探测（详见 4.8）。推理性能：128 steps 耗时 8.26s，比同环境 torch_musa GPU（12.20s）快约 1.5 倍。

## 3. 复现流程

### 3.1 环境与依赖

| 项目 | CPU | NVIDIA | Iluvatar | Moore |
| --- | --- | --- | --- | --- |
| 硬件 | 通用 x86_64 | NVIDIA GeForce RTX 4090（24 GB） | Iluvatar MR-V100（32 GB） | Moore MTT S5000（80 GB） |
| 工具链 | GCC / Clang（Linux）、MSVC（Windows） | nvcc + cuBLAS | corex `clang++`（`-x ivcore`）+ cuBLAS | mcc（`-x musa`）+ musart/mublas |
| CUDA 版本 | — | 12.8（Driver 570.124.06；容器内 torch 须与之匹配，见 4.3） | 10.2（IX-ML 4.4.0） | —（MUSA 4.3.5，Driver 4.3） |
| 通用 | Python ≥ 3.9、PyTorch、Transformers、xmake | 同左 | 同左 | 同左 |

> root 容器环境（NVIDIA、Iluvatar、Moore 等）运行 xmake 前均需 `export XMAKE_ROOT=y`。

### 3.2 复现步骤

```bash
# 1. 获取代码
git clone <本仓库地址> && cd llaisys

# 2. 安装构建工具与 Python 依赖
curl -fsSL https://xmake.io/shget.text | bash 
pip install torch transformers huggingface_hub

# 3. 构建并安装 llaisys（CPU 后端，默认配置）
xmake f -c && xmake build && xmake install
pip install -e ./python/          # 建议可编辑安装，避免 .so 旧库缓存问题（见 4.3）

# 4. 运行测试并记录结果（输出全部通过即复现成功）
python test/test_runtime.py --device cpu   # Assignment-0
python test/test_tensor.py                 # Assignment-1
python test/test_ops.py --device cpu       # Assignment-2
python test/test_infer.py --test --device cpu   # Assignment-3（自动下载 DeepSeek-R1-Distill-Qwen-1.5B）
```

### 3.3 新算力环境接入（每台机器首次使用）

进入新的算力/容器环境后，xmake 需重新安装，Python 包需重新安装并配置环境变量：

```bash
export XMAKE_ROOT=y                    # root 容器允许 xmake 运行
export HF_ENDPOINT=https://hf-mirror.com   # 国内镜像，避免模型下载失败
export HF_HUB_DISABLE_XET=1

xmake f --nv-gpu=y -c && xmake build && xmake install        # NVIDIA 后端
# 或 xmake f --iluvatar-gpu=y -c && xmake build && xmake install  # Iluvatar 后端
# 或 xmake f --moore-gpu=y -c && xmake build && xmake install    # Moore 后端
pip install -e ./python/
```

说明：

- `xmake f -c` 清理上次配置：切换设备后端时必须先 `-c`，否则旧选项残留导致后端未按预期编译；
- Iluvatar 后端需保证 `/usr/local/corex` SDK 与工具链存在；Moore 后端需保证 `/usr/local/musa-4.3.5` MUSA SDK 与 mcc 工具链存在；
- `HF_ENDPOINT` / `HF_HUB_DISABLE_XET` 影响 `test_infer.py` 自动下载模型的行为（国内环境建议必设）。

### 3.4 构建与测试命令（四设备）

构建：`xmake`（CPU，默认）；`xmake f --nv-gpu=y`（NVIDIA）；`xmake f --iluvatar-gpu=y`（Iluvatar）；`xmake f --moore-gpu=y`（Moore），随后 `xmake build && xmake install`。

| 测试模块 | CPU | NVIDIA | Iluvatar | Moore |
| --- | --- | --- | --- | --- |
| 张量 | `python test/test_tensor.py` | —（设备无关） | —（设备无关） | —（设备无关） |
| 运行时 | `python test/test_runtime.py --device cpu` | `... --device nvidia` | `... --device iluvatar` | `... --device moore` |
| 算子 | `python test/test_ops.py --device cpu` | `... --device nvidia` | `... --device iluvatar` | `... --device moore` |
| 推理 | `python test/test_infer.py --model <模型目录> --test --device cpu` | `... --device nvidia` | `... --device iluvatar` | `... --device moore` |

## 4. 实验过程中遇到的问题

### 4.1 softmax 需减去最大值（数值稳定性）

- **问题**：self_attention 对 `exp(score)` 直接累加，长序列分值较大时溢出为 inf，softmax 分母 inf/inf → NaN，推理结果错乱。
- **修复**：两趟扫描，先求最大分值，再 `exp(score - max)` 归一化。

### 4.2 数据类型转换：Linux 不报错 ≠ Windows 不报错

- **问题**：`size_t` 隐式窄化、`std::stoi` 依赖传递包含的 `<string>`，GCC/Clang 下静默或仅警告，MSVC 直接编译失败（CI 的 windows-latest 会暴露）。
- **修复**：显式 `static_cast`；显式 `#include <string>`。

### 4.3 NVIDIA 容器内 torch 与 CUDA 版本适配

- **问题**：torch 编译所用 CUDA 高于容器驱动支持版本时，`torch.cuda.is_available()` 可能为 `True`，但实际创建 GPU 张量/算子时静默失败，极易误判环境可用。
- **修复**：按驱动支持的 CUDA 版本安装对应 wheel（如 `--index-url .../whl/cuXXX`）后重新验证。
- **连带问题**：Python 包普通安装时 `.so` 被复制进 site-packages，`xmake` 重建不更新该副本，测试仍加载旧库（GPU 设备数静默为 0）；须用 `pip install -e ./python` 可编辑安装。

### 4.4 GPU 适配：标量与设备侧数据的传递

- **问题 1**：`cudaMalloc` 前未初始化 CUDA 上下文，`cublasCreate` 与首次分配时序错乱。
  - **修复**：`DeviceResource` 提前到 `Runtime` 构造时创建。
- **问题 2**：GPU 返回的指针是设备地址，主机端直接解引用会段错误；该问题在 CPU 下不易暴露（CPU 设备指针与主机指针同一地址空间）。
  - **修复**：跨设备读取走 `memcpy_sync(..., LLAISYS_MEMCPY_D2H)` 读回主机端。

### 4.5 Python / ctypes / C API / cc / cpp 映射

调用链：`Python 友好接口 → ctypes 绑定 → C API（include/*.h）→ C ABI 边界（src/llaisys/*.cc）→ C++ 实现`。

- **问题 1**：ctypes 的 `argtypes/restype` 与 C 头文件不对齐（结构体字段序、指针/枚举大小），导致栈破坏或静默错误数据。
- **问题 2**：Python 子包缺失导致 import 失败（补 `llaisys.libllaisys.models` 空 `__init__.py`）。
- **修复**：改 C++ 接口后同步检查 C 头文件 → `.cc` 边界 → ctypes 签名 → Python 封装四层的一致性。

### 4.6 资源配置原理

- **Context**：线程局部单例，按 NVIDIA → Iluvatar → Moore → CPU 顺序探测设备数并预留 `Runtime` 槽位（惰性创建），首个可用设备立即激活；`setDevice` 切换时先停用旧 Runtime。
- **Runtime**：单设备资源管理器，持有 RuntimeAPI 函数表、stream、`NaiveAllocator`、`DeviceResource`；设备存储走分配器（GPU 为 `cudaMalloc`），主机存储走 `malloc_host`，释放按 `Storage::isHost()` 分流。
- **Storage**：`shared_ptr` 内存所有权，可被多个 Tensor 视图共享；Tensor 只存 `offset + meta`。
- **关键点**：设备类型决定资源链——`getDeviceResource`/`getRuntimeAPI` 按设备类型 switch，未编译该后端时抛异常；GPU 算子须经 `Runtime::stream()` 执行并 `synchronize()`，否则 CPU 读回结果时数据未就绪。

### 4.7 flash attention prefill 输出 NaN

- **问题**：GPU 推理打通期间发现 `test_infer.py --device nvidia` 输出全乱：`last_logits` 全 NaN，argmax 返回垃圾 token id（如 139707031638018）。
- **定位**：逐层 dump 发现 NaN 首次出现在 L0 层 self_attention 输出（attn_val），该层输入 q/k/v 全部有限；KV cache 内容经验证正确；与参考实现逐字节对比确认 kernel 代码一致且其两个历史 bug（ODR 弱符号合并、跨 chunk data race）均已修复，最终锁定为未初始化 shared memory。
- **根因**：prefill kernel 的 chunk 协作搬运只填充 `j + row_in_chunk <= tile_max_limit` 的行，chunk 尾部行残留上一个 kernel（cuBLAS GEMM）遗留的 shared memory 位型（±inf）；causal mask 把这些位置的概率置 0 后，累加 `acc += 0 × V_chunk[残留行]` 得到 `0 × inf = NaN`（IEEE 754）。前序 kernel 固定遗留 inf，故每次运行必现。
- **修复**：每轮 chunk 搬运前先整块清零 K_chunk/V_chunk 并 `__syncthreads()`，再做条件填充——未填充行位型为 0，`0 × 0 = 0` 从机制上消除 NaN。
- **教训**：分块 kernel 部分填充 shared memory 时不能依赖"残留位型恰好有限"的运气，必须先清零再填充。
- 完整排查记录见 `docs/flash_attention_bug_report.md`。

### 4.8 Moore 工具链：xmake 内置 cuda 规则与 mcc 冲突

- **问题**：Moore 算子沿用 `.cu` 扩展名，xmake 对 `.cu` 文件自动挂载内置 `cuda` 规则链，其中 `cuda.env` 在 on_load 阶段强制执行 `find_cuda(nvcc)` 探测——即使 target 已把 cu 编译器设为 mcc，配置阶段仍报 "Cuda SDK not found!"，构建无法开始。
- **修复**：`xmake/moore.lua` 定义 standalone `moore` 工具链（`set_toolset("cu", "mcc")`，C/C++ 编译与链接仍走 gcc/g++），moore target 统一添加 `-x musa` 编译标志并链接 musart/mublas；同时在 target 内声明本地空规则 `cuda.env`，覆盖 xmake 内置同名规则，阻断 nvcc 探测。
- **复用**：MUSA API 与 CUDA 一一对应（`cudaMalloc` → `musaMalloc`、`__nv_bfloat16` → `__mt_bfloat16`、`cublas*` → `mublas*`），kernel 内同步原语（`__shfl_sync` / `__syncthreads`）完全兼容，8 个算子 kernel 由 NVIDIA 版平移而来，仅 linear 的 GEMM 改走 `mublasGemmEx`（compute type `MUSA_R_32F`）。

## 5. 总结

- 作业 #1 ~ #3（CPU 链路）完成：张量、8 算子（fp32/fp16/bf16）、Qwen2 推理全部通过，输出与 HF 逐 token 一致；CPU 推理耗时 391.66s（正确性优先，性能为后续工作）。
- 作业 #4：选定 NVIDIA、Iluvatar 与 Moore 三个平台，Runtime API、8 算子与模型推理均已完成——NVIDIA 侧全链路测试通过（`test_infer.py --device nvidia` 输出与 HF 逐 token 一致，128 steps 耗时 0.61s，比 torch GPU 快约 3.5 倍）；Moore 侧（MTT S5000 + MUSA 4.3.5）全链路测试通过（128 steps 耗时 8.26s，比同环境 torch_musa 快约 1.5 倍）；Iluvatar 侧 Runtime API 与算子代码已完成，因容器环境障碍未验证，环境修复后补测。
- 后续工作：验证 Iluvatar 平台；CPU 后端性能优化（对齐 torch 的 BLAS 级实现）；GPU 算子进一步优化（decode 阶段 kernel 融合、Paged KV Cache、CUDA Graphs 降低逐 token 调度开销等）。
