--!A cross-platform build utility based on Lua
--
-- mcc: Moore Threads MUSA compiler (based on clang 14)
--

-- inherit gcc (command-line compatible; avoids clang's --cuda-path handling)
-- NOTE: use absolute module name, relative import resolves to this dir
inherit("core.tools.gcc")

-- init it
function init(self)

    -- init super
    _super.init(self)

    -- shared objects need PIC for cu code
    if not self:is_plat("windows", "mingw") then
        self:add("shared.cuflags", "-fPIC")
    end

    -- suppress unused-argument warnings from forced flags
    self:add("cuflags", "-Qunused-arguments")
end
