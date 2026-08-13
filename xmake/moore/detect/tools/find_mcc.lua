--!A cross-platform build utility based on Lua
--
-- find mcc (Moore Threads MUSA compiler)
--
-- @param opt   the argument options, e.g. {version = true}
--
-- @return      program, version
--
-- @code
--
-- local mcc = find_mcc()
--
-- @endcode
--

-- imports
import("lib.detect.find_program")

-- NOTE: must NOT call find_tool() here, it would recurse back into this
-- module (find_tool imports detect.tools.find_<name> and calls main()).
function main(opt)
    return find_program("mcc", opt)
end
