target("llaisys-device-iluvatar")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("../src/device/iluvatar/*.cpp")
    add_includedirs("/usr/local/corex/include")
    add_linkdirs("/usr/local/corex/lib64")
    add_links("cudart")
    add_rpathdirs("/usr/local/corex/lib64")

    on_install(function (target) end)
target_end()
