#!/usr/bin/env python3
"""Every diagnostic the toolchain can print, with what to do about it.

The message templates are extracted from the sources (so a new `fail(...)` / `eprintln!("[MVCC] ...")` cannot go
undocumented: `--check` fails on a template without guidance); the guidance is the GUIDANCE table below, keyed by a
prefix of the template. Compiler/runtime stderr uses a `[MVCC]` prefix.

Usage: tools/gen_diagnostics.py            write the Markdown table to stdout
       tools/gen_diagnostics.py -o FILE    write it to FILE
       tools/gen_diagnostics.py --check    exit 1 if a message has no guidance (run by tests/run.sh and CI)
"""
import os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCES = {
    "compiler: device code emission ([MVCC] error: in kernel <name>: ...)": (
        ["cpp/mvcc-llvm/msl_emitter.cpp"], r'fail\("((?:[^"\\]|\\.)*)'),
    "compiler: tensor recovery ([MVCC] warning: tensor recovery: <kernel>: mma.sync recognized but not recovered ...)": (
        ["cpp/mvcc-llvm/tensor_recovery.cpp"], r'fail\("((?:[^"\\]|\\.)*)'),
    "driver ([MVCC] error: ...)": (
        ["crates/mvcc-driver/src/main.rs"], r'(?:Err\(format!\("(?:\[MVCC\] )?|Err\("\[MVCC\] |die\("\[MVCC\] )((?:[^"\\]|\\.)*)'),
    "runtime (stderr, prefixed [MVCC])": (
        ["crates/mvcc-cudart/src/api.rs", "crates/mvcc-cudart/src/module.rs", "crates/mvcc-cudart/src/runtime.rs"],
        r'eprintln!\("\[MVCC\] ((?:[^"\\]|\\.)*)'),
}

# prefix of the message template -> what it means / what to do. Longest matching prefix wins.
GUIDANCE = {
    # ---- emitter: fp64 / unsupported device features
    "double in device code": "Apple GPUs have no fp64. Use `float`; `README.md`, 'Differences from CUDA', lists the affected APIs.",
    "fp64 libdevice function __nv_": "Apple GPUs have no fp64: call the `f`-suffixed float variant.",
    "fp64 memory access in inline PTX": "`__ldcg`/`__stcg` on `double`; use the float overloads.",
    "fp literal of unsupported type": "fp64 or fp128 literal in device code; use float literals.",
    "internal: 64-bit atomic reached the 32-bit atomic path": "Report with the `.ll` (`MVCC_KEEP=1`).",
    "atomic on ": "Atomic on a type/address space Metal cannot express (e.g. 16-bit, `thread` memory). Use 32-bit device or shared atomics.",
    "float atomic on ": "Float atomics exist only for `float` on device/shared memory; atomics on `double` are not supported.",
    "bitwise atomic on float": "`atomicAnd/Or/Xor` on a float pointer; reinterpret as `unsigned` first.",
    "unsupported atomicrmw operation": "An RMW kind Metal lacks (e.g. `nand`). Use a CAS loop.",
    "function pointers are not supported on the device": "Metal has no device function pointers. Use templates or a `switch` over an enum.",
    "indirect call on device": "A call through a pointer; Metal has no device function pointers. Make the callee static.",
    "call to non-inlined function '": "Recursion or an external `__device__` function. Everything must be inlinable into the kernel; remove recursion and define the callee in the same TU.",
    "dynamic alloca": "Variable-length local arrays are not supported; use a compile-time bound.",
    "more than one extern __shared__ array": "Metal exposes one dynamic threadgroup buffer. Declare one `extern __shared__` array and carve it up.",
    "__device__ global variable '": "A module-scope `__device__`/`__constant__` variable with an initializer or type the emitter cannot place. Pass it as a kernel argument or `__constant__` scalar.",
    "unsupported global variable '": "A module-scope variable of a type/address space the emitter cannot place; pass it as a kernel argument.",
    "address space cast ": "A cast between address spaces Metal cannot express (e.g. generic -> constant). Keep pointers in one space.",
    "unsupported address space ": "Pointer in an address space with no Metal equivalent (e.g. `local` param space); restructure.",
    "unsupported integer width i": "Integers other than 1/8/16/32/64 bits (e.g. `__int128`). Use 64-bit.",
    "vector width ": "A vector type wider than Metal's 4 lanes. Use arrays or two vectors.",
    "vector.reduce on a non-fixed vector": "LLVM `vector.reduce` on a scalable or opaque vector; Metal needs a fixed width. Rewrite as a scalar loop or a fixed `<N x T>` reduction.",
    "unsupported type: ": "A device type without MSL equivalent (fp64, i128, opaque struct through a bitcast).",
    "unsupported cast": "An LLVM cast the emitter has no lowering for; report with the `.ll` (`MVCC_KEEP=1`).",
    "unsupported binary operator": "Report with the `.ll` (`MVCC_KEEP=1`).",
    "unsupported unary operator": "Report with the `.ll` (`MVCC_KEEP=1`).",
    "unsupported constant expression: ": "A constant expression (typically pointer arithmetic on a global) the emitter cannot fold; index at runtime instead.",
    "unsupported constant: ": "Report with the `.ll` (`MVCC_KEEP=1`).",
    "unsupported instruction: ": "An LLVM instruction without lowering (e.g. `invoke`, `landingpad`: exceptions do not exist on the device).",
    "unsupported intrinsic: ": "An LLVM/NVVM intrinsic outside the supported surface (`README.md`, `tools/gen_surface.py`). Report it; the lowering table is `emitIntrinsic`.",
    "unsupported libdevice function __nv_": "A math function outside the supported surface (`README.md`, `tools/gen_surface.py`). Report it; the table is `emitLibdevice`.",
    "unsupported terminator": "Control flow the structurizer cannot express (`switch` with fallthrough into a loop, `indirectbr`). Rewrite as `if`/`while`.",
    "unsupported ": "Generic: something outside the supported surface. The message names it.",
    "bad fcmp": "Report with the `.ll`.",
    "bad icmp": "Report with the `.ll`.",
    "unnamed value: ": "Internal ordering error in the emitter; report with the `.ll`.",
    "block emitted twice: ": "Internal structurizer error; report with the `.ll`.",
    "reachable block not emitted: ": "Internal structurizer error; report with the `.ll`.",
    "branch leaves loop to non-exit block ": "Irreducible control flow (a `goto` into a loop, or a `break` out of two loops at once). Restructure the loop; the state-machine fallback is used automatically when possible.",
    "conditional branch without post-dominator in block ": "Irreducible control flow; restructure the loop. A volatile wait (`grid_sync`) can make LLVM's PDT IDom the virtual root — the emitter should recover the nearest common descendant of the two arms.",
    "loop with multiple exit blocks": "A loop with `return` and `break` on different paths; the emitter unifies loop exits, and this means it could not. Hoist the `return`.",
    "unsupported terminator in loop header": "A loop header ending in `switch`/`indirectbr`; rewrite as `if`/`while`.",
    # ---- emitter: inline PTX
    "cannot parse inline PTX: ": "The asm string uses syntax outside the PTX grammar the parser accepts (`cpp/mvcc-llvm/ptx_asm.cpp`). The message carries the string.",
    "inline PTX predicate register `": "An `asm` block declares a predicate (`.reg .pred p; setp ...; @p ...`). Only the ignore-src `cp.async` form (`setp; cp.async ..., p`) is understood, as one zero-selecting copy; write other predicates as a C++ `if` around the asm.",
    "unsupported inline PTX instruction '": "A mnemonic outside the supported surface (`README.md`, `tools/gen_surface.py`). Either replace it with the CUDA intrinsic, or add the lowering (`emitInlineAsm`) with a test in `tests/kernels/intrinsics.cu`.",
    "unsupported mma variant ": "A shape/type outside the supported `mma.sync` list (`README.md`, 'Supported CUDA surface'). `mvcc::warp_tile` (`include/mvcc/tile.cuh`) is the portable alternative.",
    "unsupported cp.async form: ": "Only `cp.async.cg/ca.shared.global` with 4/8/16 bytes and optional src-size are lowered.",
    "unsupported cvt: ": "A `cvt` form outside `f16x2/bf16x2 <- f32`; use the CUDA conversion intrinsics.",
    "PTX reads write-only asm operand %": "The asm reads an `=r` operand; make it `+r`.",
    "PTX writes input operand %": "The asm writes an input operand; make it `+r` or an output.",
    "PTX shared-memory operand is a pointer to ": "An `ldmatrix`/`cp.async` shared address that does not point into `__shared__` memory (the emitter must know the address space). Pass `__cvta_generic_to_shared` of a shared array.",
    "asm operand %": "Operand number out of range for the constraint list; check the asm string.",
    "expected register operand in: ": "The instruction wants a register where the asm has an immediate or memory operand.",
    "bad ": "Malformed operand list for the named PTX instruction (count or kind); the message shows the string.",
    "cp.async needs 3 operands: ": "Malformed `cp.async`.",
    "cp.async size must be 4/8/16: ": "PTX only allows these sizes.",
    "lop3 needs 5 operands: ": "Malformed `lop3`.",
    "mma needs 4 operands: ": "Malformed `mma.sync`.",
    "ldmatrix destination count mismatch: ": "`.x1/.x2/.x4` must match the destination vector size.",
    # ---- emitter: mvcc builtins
    "__mvcc_tile_mma: ": "`mvcc::warp_tile` misuse: the accumulator must be a register array and the operands device or shared pointers (`include/mvcc/tile.cuh`).",
    "__mvcc_tp_mma: bad arity": "Internal (tensor recovery emitted a malformed op); report with the `.ll`.",
    "__mvcc_tp_ctstore: bad arity": "Internal (epilogue recovery emitted a cooperative store with the wrong value count); report with the `.ll`.",
    "unknown tensor recovery builtin ": "Internal; report.",
    "internal: ": "Internal invariant violated; report with the `.ll` (`MVCC_KEEP=1`).",
    # ---- emitter: mixed pointers (a pointer that is `thread` memory on one path and device/threadgroup on another)
    "a pointer that is thread memory on some paths and device/threadgroup memory on others": "MSL needs one address space per access; plain loads/stores through such a pointer are emitted as a select, but this operation (atomic, `memcpy`, PTX operand, ...) cannot be. Keep local scratch and device data behind distinct pointers.",
    "a pointer rebuilt from an integer slot that holds thread memory": "A pointer round-tripped through an integer (`__cvta_*`, `uintptr_t`) whose slot holds `thread` memory on some paths and device/threadgroup memory on others. Do not cast local-array addresses to integers alongside device pointers.",
    "atomic or volatile load through a mixed pointer": "Atomics/volatile accesses need one address space; give the local and the device case separate pointers.",
    "atomic or volatile store through a mixed pointer": "Atomics/volatile accesses need one address space; give the local and the device case separate pointers.",
    # ---- emitter: fp64 emulation and globals
    "fp64 operation not emulated: ": "Software binary64 covers `+ - * /`, `sqrt`, rounding, comparisons and conversions only; transcendental fp64 (`exp`, `log`, `pow`, `sin`, `cos`, `fma`) and `frem` are not emulated. Use the float versions (`README.md`, 'Semantics').",
    "constant global '": "A `__constant__`/`const` global is referenced but has no initializer in this module (declared `extern`). Define it in the same translation unit.",
    "global initializer with an integer wider than 64 bits": "An `__int128` field in a module-scope initializer; use 64-bit fields.",
    "global initializer holds a function pointer": "A table of `__device__` function pointers; Metal has none. Use a `switch` over an enum or templates.",
    "initializer points at global '": "A module-scope initializer takes the address of a global the emitter did not place in the globals buffer (e.g. a `__shared__` or `extern` symbol). Point at `__device__`/`__constant__` data defined in the same TU.",
    "initializer address not based on a global": "A module-scope initializer holds a pointer that is not `&global + constant`; compute the address at run time.",
    "initializer with a non-constant address computation": "A module-scope initializer indexes a global with a non-constant offset; compute the address at run time.",
    # ---- tensor recovery (warnings: the exact lowering is kept, the kernel is still correct)
    "A operand: ": "An `mma` A fragment's provenance is not one recovery models: `ldmatrix.x4`, two `ldmatrix.x2` fused into one block, or a register-built block whose words are plain values. The message names the reason; exact lowering runs.",
    "B operand: ": "An `mma` B fragment's provenance is not one recovery models: `ldmatrix.x4` fields, an `ldmatrix.x2` per n8 tile (two tiles sharing A fuse into one block), or register-built n8 tiles paired into 16-wide blocks. The message names the reason; exact lowering runs.",
    "A ldmatrix not recoverable: ": "The A `ldmatrix` has uses other than `mma` operands or φs.",
    "B ldmatrix not recoverable: ": "The B `ldmatrix` has uses other than `mma` operands or φs.",
    "ldmatrix feeds both A and B operands": "Unusual tile scheme; exact lowering runs.",
    "ldmatrix fields feed different A blocks in different mmas": "Inconsistent fragment use across K steps; exact lowering runs.",
    "ldmatrix fields feed different B blocks in different mma pairs": "Inconsistent B fragment use across K steps; exact lowering runs.",
    "mma result used other than by single extractvalue per field": "The accumulator is inspected mid-chain (e.g. per-step scaling). Exact lowering runs.",
    "mma variant ": "Only `m16n8k16` f16/bf16 -> f32 is recovered.",
    "no partner mma shares A and the other half of the B ldmatrix": "Recovery needs n8 pairs to form 16-wide blocks; a lone n8 tile is left exact.",
    "register-built block is shared by mma batches none of which dominates the others": "A register-built operand block (e.g. attention P) feeds `mma` batches on divergent paths, so no single point can host its recovered fill. Exact lowering runs for those batches.",
    "unexpected mma operand count": "Malformed `mma`; the exact lowering will also reject it.",
    # ---- driver
    "no input files": "Usage error.",
    "-o cannot be used with multiple input files in -c mode": "nvcc rule.",
    "could not rewrite the __shared__ declaration of '": "clang rejects `__shared__ T x;` when `T` has default member initializers; nvcc accepts it and does not run them (shared memory is uninitialized). The driver rewrites each such declaration and did not recognize this one's form (several declarators, an initializer, or a declaration split across lines). Declare the variable alone on one line as `__shared__ T name;` or `__shared__ T name[N];`, or drop the member initializers.",
    "generated Metal for ": "The MSL the compiler produced did not compile with Metal. This is a compiler bug: `MVCC_KEEP=1` keeps the `.metal`; report it with the first Metal error line.",
    # ---- runtime
    "__cudaRegisterFatBinary: bad wrapper magic": "The host object was not produced by `mvcc` (real nvcc fatbinary). Rebuild with the toolkit.",
    "cannot register device code: ": "The embedded device blob is corrupt or from an incompatible toolkit version; rebuild.",
    "cudaMallocManaged: returning pinned host memory": "Informational: managed memory is pinned host memory on unified-memory Macs; pass it to kernels directly.",
    "kernel ": "A kernel from a module whose registration failed (see the earlier error).",
    "variable <...> belongs to a module that failed to register": "A `__device__`/`__constant__` symbol from a module whose registration failed (see the earlier error); `cudaMemcpyToSymbol` on it will fail.",
    "compiled spill build of module ": "Informational (`MVCC_VERBOSE=1`): the module was recompiled with `__MVCC_SPILL=1` because a kernel's shared memory exceeds the 32 KB threadgroup limit and is served from the device-memory spill pool (`README.md`, 'Semantics').",
    "shared-memory spill pool: ": "Informational (`MVCC_VERBOSE=1`): size of the device-memory pool backing over-limit shared memory (slot count scales with core count). Slots straddling a 4 GiB boundary are unusable.",
    "spill pipeline creation failed for ": "The spill build of a kernel (shared memory above 32 KB, served from device memory) did not compile into a pipeline; the launch returns `cudaErrorInvalidKernelImage`. Reduce the kernel's shared memory, or report with `MVCC_KEEP=1` output.",
    "graph captured: ": "Informational (`MVCC_VERBOSE=1`): stream capture summary (ops, streams, dependency levels) for a `cudaGraph` built from capture.",
    "module-scope __device__/__constant__ variable '": "Unsupported by the runtime: pass the value as a kernel argument.",
    "compiled module ": "Informational (`MVCC_VERBOSE=1`): MSL compile time; `README.md`, 'Performance', for the offline path.",
    "loaded module ": "Informational: offline metallib used.",
    "module ": "Informational (`MVCC_VERBOSE=1`): the module-scope globals buffer this module was loaded with.",
    "module-scope variable '": "An `extern __device__` variable with no initializer in this translation unit; define it here or pass the value as a kernel argument.",
    "pipeline ": "Informational (`MVCC_VERBOSE=1`).",
    "tensor recovery: device has no Metal 4 TensorOps": "Pre-M5 or old macOS: exact kernels run (slower, correct).",
    "tensor recovery: exact twins selected by environment": "`MVCC_TENSOR_EXACT=1` at run time.",
    "tensor recovery: verification failed (": "The device's cooperative-tensor layout differs from the compiled law: exact kernels run. Report the message with the macOS/device.",
    "tensor recovery: ": "Informational (`MVCC_TENSOR_DIAG=1`): policy decisions and per-kernel twin fallbacks with the reason (block too large, shared memory too large, block shape).",
    "check+emit kept: L1": "Catalog miss: the recovered Program has no legal covering of (o,r). Exact twin stays. Closed vocab: catalog miss / L1.",
    "check+emit kept: L5": "Effect order: an observable store or atomic would fire more than once. Exact twin stays.",
    "check+emit kept: L7": "Numerical class: raw mma.sync was not recovered. Exact twin stays.",
    "check+emit kept: L10": "An assume (non-affine index) cannot be discharged at compile time. Exact twin stays.",
    "relayout deleted kernel": "L12: a layout-conversion kernel was deleted by the schedule (producer/consumer already agree; not host-visible).",
    "legality: L11": "Consumer block would read bytes another block wrote. Metal has no grid barrier; refuse the fusion.",
    "legality: L12": "Host-visible pointer aliases an intermediate; layout is not free.",
    "legality: L13": "Silent quantization. Default is inherit; pass an explicit budget.",
    "precision explicit-budget": "Width may drop only with a static error bound and a differential check against the exact twin.",
    "<...>: <...> (falling back to MSL source)": "Offline metallib (`MVCC_METALLIB_DIR`) could not be loaded; the runtime compiled from source instead.",
    "GPU fault: ": "A kernel faulted (out-of-bounds access, most often). Subsequent CUDA calls return `cudaErrorIllegalAddress`. Run with a smaller problem and check indexing; Metal's shader validation (`MTL_SHADER_VALIDATION=1`) points at the store.",
    "device ": "Informational (`MVCC_VERBOSE=1`): device summary.",
    "launch of ": "Launch rejected: the message says whether it is shared memory (32 KB limit on Apple GPUs) or threads per block (pipeline `maxTotalThreadsPerThreadgroup`, lowered by register pressure). Reduce the tile / block size; `cudaOccupancyMaxActiveBlocksPerMultiprocessor` reports the limits.",
    "pipeline creation failed for ": "Metal rejected the kernel at pipeline creation (usually a resource limit). Report with `MVCC_KEEP=1` output.",
    "prewarm failed: ": "Background pipeline compilation failed; the launch will report the real error.",
    "registered module ": "Informational (`MVCC_VERBOSE=1`).",
    "emitted collective sequence: ": "Informational: the module's ABI names the NCCL sequence `bind:device` emitted (AllGather copy; AllReduce / Reduce / ReduceScatter are gather-and-sum or avg of f32/i32). Prod / max / min still return `ncclInvalidUsage`.",
    "usage: mvcc diff ": "`mvcc diff` takes two schedule-text files (`mvcc-ir2msl --sched-diff`).",
    "usage: mvcc stir-in ": "`mvcc stir-in` constructs a STIR Program without CUDA recovery (`elementwise|gemm|attention` or a `.stir` file; optional `--sched-in`).",
    "<...>": "Verbatim error text from the runtime's registration path (`cudaGetErrorString` carries the same).",
}


def extract():
    out = {}
    for section, spec in SOURCES.items():
        msgs = set()
        specs = spec if isinstance(spec, list) else [spec + ("",)]
        for files, pat, prefix in specs:
            for f in files:
                src = open(os.path.join(ROOT, f)).read()
                for m in re.findall(pat, src):
                    m = m.replace("\\\"", "\"").replace("\\n", " ").strip()
                    if section.startswith("runtime") or section.startswith("driver"):
                        m = re.sub(r"\{[^}]*\}", "<...>", m)
                    if m:
                        msgs.add(prefix + m)
        out[section] = sorted(msgs)
    return out


def guidance_for(msg):
    best = None
    for k in GUIDANCE:
        if msg.startswith(k.rstrip()) and (best is None or len(k) > len(best)):
            best = k
    return best


def render(msgs):
    L = []
    w = L.append
    w("# Diagnostics")
    w("")
    w("Every message the toolchain prints, with what to do. Generated by `tools/gen_diagnostics.py` from the sources")
    w("(templates) and its guidance table; `tests/run.sh` fails when a message is undocumented.")
    w("")
    w("Compiler errors are hard errors by design: an unsupported construct names itself and stops the build rather than")
    w("producing a kernel that is silently wrong. Tensor-recovery messages are warnings: the exact lowering is kept and the")
    w("kernel is correct, only slower. Runtime messages go to stderr with a `[MVCC]` prefix; informational ones appear only")
    w("under `MVCC_VERBOSE=1` / `MVCC_TENSOR_DIAG=1`.")
    w("")
    w("General tools: `MVCC_KEEP=1` keeps `.ll`, `.metal` and `.abi.json` next to the output; `MVCC_TENSOR_DIAG=1` explains")
    w("recovery decisions; `MVCC_VERBOSE=1` prints every sub-command (compiler) and every module/pipeline event (runtime).")
    w("")
    missing = []
    for section, lst in msgs.items():
        w("## %s" % section)
        w("")
        w("| message | meaning / what to do |")
        w("|---|---|")
        for m in lst:
            k = guidance_for(m)
            if k is None:
                missing.append((section, m))
                g = "**undocumented**"
            else:
                g = GUIDANCE[k]
            w("| `%s` | %s |" % (m.replace("|", "\\|"), g.replace("|", "\\|")))
        w("")
    return "\n".join(L), missing


def main():
    text, missing = render(extract())
    if missing:
        for s, m in missing:
            sys.stderr.write("no guidance for [%s] %r\n" % (s, m))
        sys.exit(1)
    if "--check" in sys.argv:
        print("every diagnostic has guidance")
        return
    if "-o" in sys.argv:
        out = sys.argv[sys.argv.index("-o") + 1]
        open(out, "w").write(text)
        print("wrote %s" % out)
        return
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
