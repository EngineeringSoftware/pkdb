import gdb
import os
import re


def _hs_wrapper_info(kernel_name):
    out = gdb.execute("info address " + kernel_name, to_string=True)
    name_m = re.search(r'Symbol\s+"([^"]+)"', out)
    addr_m = re.search(r"address\s+(0x[0-9a-fA-F]+)", out)
    if not addr_m:
        raise gdb.error("Cannot resolve address of '%s': %s" % (kernel_name, out.strip()))
    name = name_m.group(1) if name_m else kernel_name
    return name, int(addr_m.group(1), 16)


def _hs_stub_addr(kernel_name):
    resolved_name, wrapper_addr = _hs_wrapper_info(kernel_name)
    try:
        arch = gdb.selected_frame().architecture()
    except gdb.error:
        objs = gdb.objfiles()
        arch = objs[0].architecture() if objs else None
    if arch is None:
        gdb.write("[hotswap] warning: no architecture, falling back to wrapper\n")
        return resolved_name, wrapper_addr
    insns = arch.disassemble(wrapper_addr, wrapper_addr + 64)
    for insn in reversed(insns):
        asm = insn["asm"]
        if re.match(r"\s*call", asm):
            m = re.search(r"0x([0-9a-fA-F]+)", asm)
            if m:
                return resolved_name, int(m.group(1), 16)
    gdb.write("[hotswap] warning: no call found in wrapper, using wrapper address\n")
    return resolved_name, wrapper_addr


def _ptx_entries(ptx_path):
    path = os.path.expanduser(ptx_path)
    if not os.path.isfile(path):
        raise gdb.GdbError("PTX file not found: %s" % path)
    entries = []
    with open(path, "r") as f:
        for line in f:
            m = re.search(r"\.entry\s+([A-Za-z0-9_]+)", line)
            if m:
                mangled = m.group(1)
                try:
                    out = gdb.execute("shell c++filt " + mangled, to_string=True)
                    if out:
                        demangled = out.strip().split("\n")[-1].strip()
                        if demangled.startswith("(gdb)"):
                            demangled = demangled[5:].strip()
                    else:
                        demangled = mangled
                except Exception:
                    demangled = mangled
                entries.append((mangled, demangled))
    return entries


def _frame_eval(names, default=None):
    for name in names:
        try:
            val = gdb.parse_and_eval(name)
            if val is not None:
                return val
        except gdb.error:
            continue
    return default


def _int_or_default(val, lo, hi, default):
    try:
        n = int(val)
        if lo <= n <= hi:
            return n
    except (gdb.error, TypeError, ValueError):
        pass
    return default


def _uk_stub_addr(kernel_name):
    out = gdb.execute("info address " + kernel_name, to_string=True)
    addr_m = re.search(r"address\s+(0x[0-9a-fA-F]+)", out)
    if not addr_m:
        return None
    wrapper_addr = int(addr_m.group(1), 16)
    try:
        arch = gdb.selected_frame().architecture()
    except gdb.error:
        arch = gdb.objfiles()[0].architecture() if gdb.objfiles() else None
    if not arch:
        return wrapper_addr
    insns = arch.disassemble(wrapper_addr, wrapper_addr + 64)
    for insn in reversed(insns):
        if re.match(r"\s*call", insn["asm"]):
            m = re.search(r"0x([0-9a-fA-F]+)", insn["asm"])
            if m:
                return int(m.group(1), 16)
    return wrapper_addr


class _HsSwapBP(gdb.Breakpoint):
    def __init__(self, from_addr, to_addr, to_name):
        super().__init__("*0x%x" % from_addr, temporary=True, internal=False)
        self._ta = to_addr
        self._tn = to_name

    def stop(self):
        gdb.execute("set $pc = 0x%x" % self._ta, to_string=True)
        gdb.write("[hotswap] kernel swapped -> %s\n" % self._tn)
        bp = self
        gdb.post_event(lambda: bp.delete())
        return False


class Hotswap(gdb.Command):
    def __init__(self):
        super().__init__("hotswap", gdb.COMMAND_USER)

    def invoke(self, argument, from_tty):
        argv = gdb.string_to_argv(argument)
        if len(argv) != 2:
            gdb.write("Usage: hotswap <from_kernel> <to_kernel>\n")
            return
        from_k, to_k = argv
        try:
            from_name, from_stub = _hs_stub_addr(from_k)
            to_name, to_stub = _hs_stub_addr(to_k)
        except gdb.error as e:
            gdb.write("[hotswap] error: %s\n" % e)
            return
        gdb.write("[hotswap] armed: %s -> %s\n" % (from_name, to_name))
        gdb.write("[hotswap] stubs: 0x%x -> 0x%x\n" % (from_stub, to_stub))
        _HsSwapBP(from_stub, to_stub, to_name)
        gdb.write("[hotswap] Swap armed. Use 'continue' or 'run' when ready.\n")


class HotswapEscape(gdb.Command):
    def __init__(self):
        super().__init__("hotswap-escape", gdb.COMMAND_USER)

    def invoke(self, argument, from_tty):
        argv = gdb.string_to_argv(argument)
        if len(argv) != 1:
            gdb.write("Usage: hotswap-escape <line_number>\n")
            return
        line = argv[0]
        try:
            gdb.execute("delete breakpoints", to_string=True)
            gdb.execute("tbreak main.cu:%s" % line, to_string=True)
            gdb.execute("continue", to_string=True)
        except gdb.error as e:
            gdb.write("[hotswap-escape] error: %s\n" % e)


class UkRedirectBP(gdb.Breakpoint):
    def __init__(self, addr, ptx_path, mangled):
        super().__init__("*0x%x" % addr, temporary=False, internal=False)
        self.ptx_path = ptx_path.replace("\\", "\\\\").replace('"', '\\"')
        self.mangled = mangled

    def stop(self):
        gdb.execute(
            'call (int)hs_load("%s", "%s")' % (self.ptx_path, self.mangled),
            to_string=True,
        )
        gdb.execute("call (int)hs_arg_reset()", to_string=True)
        try:
            view = int(gdb.parse_and_eval("$rdi"))
            size = int(gdb.parse_and_eval("$rsi"))
        except (gdb.error, TypeError, ValueError):
            return True

        gdb.execute("call (int)hs_arg_ptr((void*)%s)" % view, to_string=True)
        gdb.execute("call (int)hs_arg_int(%s)" % size, to_string=True)

        gx, bx = 1, 256
        try:
            cur = gdb.selected_frame()
            cf = gdb.newest_frame().older()
            if cf:
                cf = cf.older()
            if cf:
                cf.select()
                gx = _frame_eval(["blocksPerGrid", "gx"], 1)
                bx = _frame_eval(["threadsPerBlock", "bx"], 256)
                gx = int(gx) if gx is not None else 1
                bx = int(bx) if bx is not None else 256
                cur.select()
        except Exception:
            pass

        gdb.execute("call (int)hs_exec(%u, %u)" % (gx, bx), to_string=True)
        gdb.execute("call (int)hs_clear_err()", to_string=True)

        try:
            rsp = int(gdb.parse_and_eval("$rsp"))
            ret = int.from_bytes(gdb.selected_inferior().read_memory(rsp, 8).tobytes(), "little")
            gdb.execute("set $pc = (void*)0x%x" % ret, to_string=True)
        except Exception:
            pass

        gdb.write("[update_kernels] ran PTX %s (grid=%s, block=%s)\n" % (self.mangled, gx, bx))
        return False


class UpdateKernels(gdb.Command):
    def __init__(self):
        super().__init__("update_kernels", gdb.COMMAND_USER)

    def invoke(self, argument, from_tty):
        argv = gdb.string_to_argv(argument)
        if not argv:
            gdb.write("update-kernels: usage: update-kernels <ptx_path> [kernel_name]\n")
            return

        ptx_path = argv[0]
        kernel_name = argv[1] if len(argv) >= 2 else None

        try:
            entries = _ptx_entries(ptx_path)
            if not entries:
                gdb.write("update-kernels: no .entry found in %s\n" % ptx_path)
                return

            mangled = None
            if kernel_name:
                kn = kernel_name.strip()
                for m_name, demangled in entries:
                    if m_name == kn or demangled == kn or demangled.startswith(kn + "("):
                        mangled = m_name
                        break
                if mangled is None:
                    gdb.write(
                        "update-kernels: no kernel matching '%s'. Entries: %s\n"
                        % (kernel_name, ", ".join(d for _, d in entries))
                    )
                    return
            else:
                mangled = entries[0][0]
                if len(entries) > 1:
                    gdb.write("update-kernels: multiple kernels, using first: %s\n" % entries[0][1])

            # Ensure driver context exists; actual loads happen in UkRedirectBP.
            gdb.execute("call (int)hs_init()", to_string=True)

            # With a specific kernel name: redirect only that kernel.
            if kernel_name and kernel_name.strip():
                stub_addr = _uk_stub_addr(kernel_name)
                if stub_addr is not None:
                    UkRedirectBP(stub_addr, ptx_path, mangled)
                    gdb.write("update-kernels: %s replaced by PTX; program launches will use it.\n" % kernel_name)
                return

            # Without a kernel name: try to redirect all visible kernels from this PTX.
            for m_name, demangled in entries:
                short = demangled.split("(", 1)[0]
                for kn in (short, m_name):
                    if not kn:
                        continue
                    try:
                        stub_addr = _uk_stub_addr(kn)
                    except gdb.error:
                        stub_addr = None
                    if stub_addr is not None:
                        UkRedirectBP(stub_addr, ptx_path, m_name)
                        gdb.write("update-kernels: %s replaced by PTX %s; launches will use PTX.\n" % (kn, m_name))
                        break
        except Exception as e:
            gdb.write("update-kernels: error: %s\n" % str(e))


Hotswap()
HotswapEscape()
UpdateKernels()
