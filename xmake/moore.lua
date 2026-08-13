toolchain("moore")
    set_kind("standalone")
    set_bindir("/usr/local/musa-4.3.5/bin")

    -- 编译 .cu 文件
    set_toolset("cu", "mcc")

    -- 编译普通 C 文件
    set_toolset("cc", "/usr/bin/gcc")

    -- 编译普通 C++ 文件
    set_toolset("cxx", "/usr/bin/g++")

    -- 链接 C++ 目标
    set_toolset("ld", "/usr/bin/g++")
    set_toolset("sh", "/usr/bin/g++")

toolchain_end()

-- Replace xmake's built-in cuda.env rule with a no-op: the built-in one
-- probes for a CUDA SDK (find_cuda/nvcc) and injects -rdc=true plus
-- cudadevrt/cudart_static links, none of which apply to mcc (MUSA).
-- Target rules already present are kept when the cuda language attaches
-- its rule chain to .cu files, so adding this rule to the moore targets
-- prevents the built-in cuda.env from ever loading.
rule("cuda.env")

target("llaisys-device-moore")
    set_kind("static")
    set_toolchains("moore")
    set_languages("cxx17")
    add_rules("cuda.env")
    add_cuflags("-x", "musa", {force = true})
    add_cuflags("-std=c++17", {force = true})
    add_cuflags("-fPIC", {force = true})

    add_files("../src/device/moore/*.cu")
    add_includedirs("/usr/local/musa-4.3.5/include")
    add_links("musart")
    add_links("mublas")
    add_linkdirs("/usr/local/musa-4.3.5/lib")
    on_install(function (target) end)
target_end()

target("llaisys-ops-moore")
    set_kind("static")
    set_toolchains("moore")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    add_rules("cuda.env")
    add_cuflags("-x", "musa", {force = true})
    add_cuflags("-std=c++17", {force = true})
    add_cuflags("-fPIC", {force = true})

    add_files("../src/ops/*/moore/*.cu")
    add_includedirs("/usr/local/musa-4.3.5/include")
    add_links("musart")
    add_links("mublas")
    add_linkdirs("/usr/local/musa-4.3.5/lib")
    add_rpathdirs("/usr/local/musa-4.3.5/lib")

    on_install(function (target) end)
target_end()
