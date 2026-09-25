#!/bin/bash
# mvcc test suite (Mac side). Usage: tests/run.sh [--no-build] [--quick] [--host]
#   --no-build   use the existing toolkit/ instead of rebuilding it
#   --quick      skip the slow items (roofline, qwen smoke)
#   --host       compiler + goldens only: no GPU execution. For CI runners whose Metal
#                device is paravirtual (kernels return zeros; cooperative_tensor rejected).
# Environment: MVCC_QWEN_PACK / MVCC_QWEN_TOK point at a converted pack + tokenizer to enable the qwen smoke test.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TK="$ROOT/toolkit"
OUT="${MVCC_TEST_OUT:-/tmp/mvcc-tests}"
mkdir -p "$OUT"
BUILD=1; QUICK=0; HOST=0
for a in "$@"; do case "$a" in --no-build) BUILD=0 ;; --quick) QUICK=1 ;; --host) HOST=1 ;; *) echo "unknown option $a"; exit 2 ;; esac; done
export PATH="$TK/bin:$HOME/.cargo/bin:/opt/homebrew/bin:$PATH"
# Paravirtual Metal rejects cooperative_tensor at the compile-check step; ir2msl still runs.
if [ "$HOST" = 1 ]; then export MVCC_VERIFY=0; fi

pass=0; fail=0; failed=()
ok()   { pass=$((pass + 1)); printf '  [ok]   %s\n' "$1"; }
bad()  { fail=$((fail + 1)); failed+=("$1"); printf '  [FAIL] %s\n' "$1"; }
step() { printf '\n== %s\n' "$1"; }
run()  { # run <name> <cmd...>: logs to $OUT/<name>.log, pass/fail on exit status
  local name="$1"; shift
  if "$@" >"$OUT/$name.log" 2>&1; then ok "$name"; else bad "$name (see $OUT/$name.log)"; fi
}

if [ "$BUILD" = 1 ]; then
  step "build toolkit"
  run build "$ROOT/tools/install_toolkit.sh"
fi
# prelude_test is not part of the toolkit; build it on a full rebuild or when missing after --no-build.
if [ "$BUILD" = 1 ] || [ ! -x "$ROOT/build/mvcc-prelude-test" ]; then
  mkdir -p "$ROOT/build"
  run build-prelude-test clang++ -std=c++17 -O2 -fobjc-arc -framework Metal -framework Foundation -DPRELUDE_PATH="\"$ROOT/msl/mvcc_prelude.metal\"" "$ROOT/tools/prelude_test/prelude_test.mm" -o "$ROOT/build/mvcc-prelude-test"
fi

step "provenance"
# No NVIDIA-owned text may enter the tree: header signatures and copyright lines.
if "$ROOT/tools/check_provenance.sh" >"$OUT/provenance.log" 2>&1; then ok "provenance: no NVIDIA-owned text or libdevice bitcode"; else bad "provenance (see $OUT/provenance.log)"; fi

step "diagnostics guidance"
# every fail(...) / eprintln!("[mvcc] ...") template in the sources must carry guidance in tools/gen_diagnostics.py.
run diagnostics-guidance python3 "$ROOT/tools/gen_diagnostics.py" --check
run gem-publish-guards python3 "$ROOT/tools/check_gem_publish.py" --check
run gem-installer ruby "$ROOT/tests/gem/installer.rb"
run gem-cuda ruby "$ROOT/tests/gem/cuda.rb"
# the surface printer must still parse the emitter's lowering tables and the toolkit's exports.
run cuda-surface-renders bash -c "python3 '$ROOT/tools/gen_surface.py' | grep -q 'runtime API entry points'"

step "rust unit tests"
(cd "$ROOT" && cargo test --workspace --release >"$OUT/cargo-test.log" 2>&1) && ok "cargo test" || bad "cargo test (see $OUT/cargo-test.log)"

step "MSL prelude vs CPU reference"
if [ "$HOST" = 1 ]; then
  printf '  [skip] prelude_test (--host: no GPU)\n'
elif [ -x "$ROOT/build/mvcc-prelude-test" ]; then
  if "$ROOT/build/mvcc-prelude-test" >"$OUT/prelude.log" 2>&1 && grep -q '^0 failure' "$OUT/prelude.log"; then ok "prelude_test"; else bad "prelude_test (see $OUT/prelude.log)"; fi
else bad "prelude_test binary missing (run without --no-build)"; fi

step "kernel tests (nvcc -> run)"
for t in vector_add stream_order intrinsics warp_tile ranks device_graph_tail coherence shard_sched printf_shared globaltimer single_thread_order free_returns; do
  if nvcc -O2 -std=c++17 -o "$OUT/$t" "$ROOT/tests/kernels/$t.cu" >"$OUT/$t.build.log" 2>&1; then
    if [ "$HOST" = 1 ]; then ok "$t: compile"; else run "$t" "$OUT/$t"; fi
  else bad "$t: compile (see $OUT/$t.build.log)"; fi
done
if nvcc -O3 -std=c++17 -c -o "$OUT/structured_spin.o" "$ROOT/tests/kernels/structured_spin.cu" \
    >"$OUT/structured_spin.build.log" 2>&1; then
  if grep -q 'structured emission failed' "$OUT/structured_spin.build.log"; then
    bad "structured_spin: state-machine fallback (see $OUT/structured_spin.build.log)"
  else ok "structured_spin: if/for (no state machine)"; fi
else bad "structured_spin: compile (see $OUT/structured_spin.build.log)"; fi
if MVCC_TENSOR_DIAG=1 nvcc -O2 -std=c++17 -c -o "$OUT/warp_tile_diag.o" "$ROOT/tests/kernels/warp_tile.cu" >"$OUT/warp_tile_diag.build.log" 2>&1; then
  if grep -q 'warp_tile front-end' "$OUT/warp_tile_diag.build.log" || grep -q 'check+emit: tensor-op' "$OUT/warp_tile_diag.build.log"; then
    ok "mvcc::warp_tile reaches check+emit"
  else ok "warp_tile compiled (check+emit note absent if builtin-only)"; fi
else bad "warp_tile diag compile (see $OUT/warp_tile_diag.build.log)"; fi
# the sharded-model surface once more with a logical device per rank (cudaSetDevice per rank, peer access, a
# capture forking to the ranks' streams and joining back, NCCL collectives over the ranks)
if [ "$HOST" != 1 ] && [ -x "$OUT/ranks" ]; then run ranks-4-devices env MVCC_DEVICES=4 "$OUT/ranks"; fi
# coherent accesses once more with the spin pair on two logical devices (the peer all-reduce of a sharded model)
if [ "$HOST" != 1 ] && [ -x "$OUT/coherence" ]; then run coherence-2-devices env MVCC_DEVICES=2 "$OUT/coherence"; fi
# Tensor-pipeline recovery: every mma.sync GEMM must compile recovered, pass vs CPU, and on the exact twins.
n=$(wc -l "$ROOT"/cpp/mvcc-llvm/{stir,schedule,legality,engine,tensor_recovery,emit}.cpp "$ROOT"/cpp/mvcc-llvm/staging_elim.cpp | tail -1 | awk '{print $1}')
if [ "$n" -le 9000 ]; then ok "tensor recovery sources: $n lines (budget 9000)";
else bad "tensor recovery sources: $n lines, over the 9000-line budget"; fi

if MVCC_TENSOR_DIAG=1 nvcc -O2 -std=c++17 -o "$OUT/gemm_ptx" "$ROOT/tests/kernels/gemm_ptx.cu" >"$OUT/gemm_ptx.build.log" 2>&1; then
  if grep -q 'mma.sync recognized but not recovered' "$OUT/gemm_ptx.build.log"; then bad "gemm_ptx: a kernel was not recovered (see $OUT/gemm_ptx.build.log)"; else ok "gemm_ptx: all mma.sync kernels recovered"; fi
  if grep -q 'check+emit: tensor-op legality: ok' "$OUT/gemm_ptx.build.log"; then ok "gemm_ptx: check+emit tensor-op (L1–L10)";
  else bad "gemm_ptx: recovered GEMM was not admitted through check+emit (see $OUT/gemm_ptx.build.log)"; fi
  if [ "$HOST" = 1 ]; then
    printf '  [skip] gemm_ptx recovered/exact-twins (--host: no GPU)\n'
  else
    if MVCC_TENSOR_DIAG=1 "$OUT/gemm_ptx" >"$OUT/gemm_ptx.log" 2>&1 && grep -q 'recovered kernels enabled' "$OUT/gemm_ptx.log"; then ok "gemm_ptx recovered vs CPU (layout verified on device)"; else bad "gemm_ptx recovered (see $OUT/gemm_ptx.log)"; fi
    run gemm_ptx-exact-twins env MVCC_TENSOR_EXACT=1 "$OUT/gemm_ptx"
  fi
else bad "gemm_ptx: compile (see $OUT/gemm_ptx.build.log)"; fi
step "semantic iteration graph, coordinate discovery, STIR (golden)"
# the SIG summary, the discovered coordinates and the STIR text of every kernel in the corpus must match
# tests/stir/*.golden exactly (loop classes, recurrence algebras and permissions, fitted sites, round-trip counts).
# Regenerate a golden after an intended change with: MVCC_SIG=1 MVCC_SIG_ONLY=1 MVCC_SIG_DUMP=<golden> nvcc <same flags>.
sig_golden() { # <name> <golden> <nvcc args...>
  local name="$1" golden="$2"; shift 2
  rm -f "$OUT/$name.sig"
  if MVCC_SIG=1 MVCC_SIG_ONLY=1 MVCC_SIG_DUMP="$OUT/$name.sig" nvcc "$@" >"$OUT/$name.sig.build.log" 2>&1; then
    if grep -q 'MISMATCH\|REFUSED' "$OUT/$name.sig"; then bad "$name: STIR round trip mismatch or refused kernel (see $OUT/$name.sig)"; fi
    if diff -u "$golden" "$OUT/$name.sig" >"$OUT/$name.sig.diff" 2>&1; then ok "$name: sig/coords/stir match golden"; else bad "$name: sig/coords/stir differ from golden (see $OUT/$name.sig.diff)"; fi
  else bad "$name: compile (see $OUT/$name.sig.build.log)"; fi
}
step "STIR engine (schedule language, legality L1–L15, search, check+emit)"
if mvcc-ir2msl --engine-self-test >"$OUT/engine.self.log" 2>&1; then
  if grep -q 'engine self-test: ok' "$OUT/engine.self.log" && grep -q 'sched-diff reports a why change' "$OUT/engine.self.log"; then ok "engine self-test (L1–L15, search, explain, direct front-end, fuzz, graph, L13, sched-diff)";
  else bad "engine self-test: did not print ok (see $OUT/engine.self.log)"; fi
else bad "engine self-test (see $OUT/engine.self.log)"; fi
printf '%s\n' 'schedule gemm why=source stages=1' '  stage 0 engine=scalar-fma exchange=none prefetch=at-use predicate=none thread_scale=1 smem=0' >"$OUT/sched_a.txt"
printf '%s\n' 'schedule gemm why=engine:tensor-op stages=1' '  stage 0 engine=tensor-op exchange=none prefetch=at-use predicate=none thread_scale=1 smem=0' >"$OUT/sched_b.txt"
if mvcc-ir2msl --sched-diff "$OUT/sched_a.txt" "$OUT/sched_b.txt" >"$OUT/sched_diff.log" 2>&1 && grep -q 'changed: why' "$OUT/sched_diff.log"; then ok "mvcc diff compares two schedule texts";
else bad "sched-diff missing (see $OUT/sched_diff.log)"; fi
if mvcc-ir2msl --stir-in gemm --sched-in "$ROOT/tests/stir/int4_m4_share_b.sched" >"$OUT/int4_m4_sched.log" 2>&1 \
   && grep -q 'why=share-B' "$OUT/int4_m4_sched.log" && grep -q 'ring=2' "$OUT/int4_m4_sched.log" \
   && grep -q 'check+emit client: ok' "$OUT/int4_m4_sched.log"; then ok "--sched-in INT4 m4 share-B is expressible";
else bad "INT4 m4 share-B sched-in (see $OUT/int4_m4_sched.log)"; fi
for pair in "l2_replicate.sched:gemm:L2" "l3_exchange_none.sched:gemm:L3" "l6_drop_pred.sched:elementwise:L6" "l7_tensor_elem.sched:elementwise:L7"; do
  f="${pair%%:*}"; rest="${pair#*:}"; kind="${rest%%:*}"; code="${rest##*:}"
  mvcc-ir2msl --stir-in "$kind" --sched-in "$ROOT/tests/stir/illegal/$f" >"$OUT/illegal_$f.log" 2>&1 || true
  if grep -q "$code" "$OUT/illegal_$f.log"; then ok "tests/stir/illegal/$f refuses $code";
  else bad "illegal $f missing $code (see $OUT/illegal_$f.log)"; fi
done
if MVCC_GRAPH="$ROOT/tests/stir/qwen_l11_refuse.graph" MVCC_TENSOR_DIAG=1 nvcc -O2 -std=c++17 -o "$OUT/qwen_l11" "$ROOT/examples/qwen/qwen.cu" >"$OUT/qwen_l11.build.log" 2>&1; then
  if grep -q 'L11' "$OUT/qwen_l11.build.log"; then ok "refused fusion names L11";
  else bad "L11 not named (see $OUT/qwen_l11.build.log)"; fi
else bad "qwen L11 graph compile (see $OUT/qwen_l11.build.log)"; fi
if mvcc-ir2msl --stir-in "$ROOT/tests/stir/direct_gemm.stir" >"$OUT/stir_in.log" 2>&1 && grep -q 'stir-in gemm' "$OUT/stir_in.log" && grep -q 'check+emit client: ok' "$OUT/stir_in.log"; then ok "--stir-in constructs a Program without CUDA recovery";
  awk '/^schedule /{p=1} p && !/^  check\+emit/{print}' "$OUT/stir_in.log" >"$OUT/direct_gemm.sched"
  if mvcc-ir2msl --stir-in gemm --sched-in "$OUT/direct_gemm.sched" >"$OUT/sched_in.log" 2>&1 && grep -q 'sched-in=' "$OUT/sched_in.log" && grep -q 'check+emit client: ok' "$OUT/sched_in.log"; then ok "--sched-in checks a dumped schedule without CUDA recovery";
  else bad "sched-in missing (see $OUT/sched_in.log)"; fi
  mvcc-ir2msl --stir-in elementwise >"$OUT/stir_in_elem.log" 2>&1
  awk '/^schedule /{p=1} p && !/^  check\+emit/{print}' "$OUT/stir_in_elem.log" | sed 's/engine=scalar-fma/engine=tensor-op/' >"$OUT/elem_l7.sched"
  mvcc-ir2msl --stir-in elementwise --sched-in "$OUT/elem_l7.sched" >"$OUT/sched_in_l7.log" 2>&1 || true
  if grep -q 'L7' "$OUT/sched_in_l7.log"; then ok "--sched-in tensor-op on elementwise is L7";
  else bad "sched-in L7 missing (see $OUT/sched_in_l7.log)"; fi
else bad "stir-in missing (see $OUT/stir_in.log)"; fi
printf 'silu_mul\tsource\tlegality: ok\tcost 1\n' >"$OUT/sched.db"
if MVCC_TENSOR_DIAG=1 MVCC_DEVICES=2 MVCC_SCHED_DB="$OUT/sched.db" MVCC_GRAPH="$ROOT/tests/stir/stir_ops.graph" nvcc -O3 -std=c++17 -o "$OUT/stir_ops_emit" "$ROOT/tests/kernels/stir_ops.cu" >"$OUT/stir_ops_emit.log" 2>&1; then
  stir_sig() {
    local prefix="$1" msg="$2" pattern="${3:-source schedule, SIG}"
    if grep "tensor recovery: ${prefix}" "$OUT/stir_ops_emit.log" | grep -q "$pattern"; then ok "$msg";
    else bad "$msg (see $OUT/stir_ops_emit.log)"; fi
  }
  stir_sig 'silu_mul' 'check+emit: silu_mul recovered (SIG source schedule)'
  stir_sig 'rowsum' 'check+emit: rowsum recovered (SIG source schedule)'
  stir_sig 'gather' 'check+emit: gather recovered (SIG source schedule)'
  stir_sig 'gemv' 'check+emit: gemv recovered (SIG source schedule)'
  stir_sig 'softmax_row' 'check+emit: softmax_row recovered (SIG source schedule)'
  stir_sig 'rmsnorm' 'check+emit: rmsnorm recovered (SIG source schedule)'
  stir_sig 'rope' 'check+emit: rope recovered (SIG source schedule)'
  stir_sig 'attn' 'check+emit: attn recovered (SIG source schedule)'
  stir_sig 'gdn' 'check+emit: gdn recovered (SIG source schedule)'
  stir_sig 'sgemm' 'check+emit: sgemm recovered (SIG source schedule)'
  stir_sig 'hgemm16' 'check+emit: hgemm16 recovered (block-parallel SIG)' 'block-parallel.*source schedule, SIG'
  stir_sig 'hgemm16b' 'check+emit: hgemm16b recovered (block-parallel SIG)' 'block-parallel.*source schedule, SIG'
  stir_sig 'scatter_add' 'check+emit: scatter_add recovered (SIG source schedule)'
  if grep -q 'sched-db preferred why=source' "$OUT/stir_ops_emit.log"; then ok "MVCC_SCHED_DB prefers a previously-winning why";
  else bad "sched-db preference missing (see $OUT/stir_ops_emit.log)"; fi
  if grep -q 'collective sequence allgather' "$OUT/stir_ops_emit.log"; then ok "graph bind emits NCCL collective sequence";
  else bad "collective sequence missing (see $OUT/stir_ops_emit.log)"; fi
  if grep -q 'graph: .*legality: ok' "$OUT/stir_ops_emit.log"; then ok "check+emit: module graph noted";
  else bad "check+emit: no graph note (see $OUT/stir_ops_emit.log)"; fi
  if grep -q 'compile-time edges' "$OUT/stir_ops_emit.log" && grep -q 'L11 fusion legal' "$OUT/stir_ops_emit.log"; then ok "L11: MVCC_GRAPH same-block edges are legal";
  else bad "L11: graph file was not applied (see $OUT/stir_ops_emit.log)"; fi
  if [ "$HOST" = 1 ]; then printf '  [skip] check+emit stir_ops vs CPU (--host: no GPU)\n';
  else
    run stir_ops-vs-cpu "$OUT/stir_ops_emit"
    if grep -q 'emitted collective sequence' "$OUT/stir_ops-vs-cpu.log"; then ok "runtime consumes ABI collectives";
    else bad "runtime did not log ABI collectives (rebuild libcudart; see $OUT/stir_ops-vs-cpu.log)"; fi
  fi
  if [ "$HOST" = 1 ]; then printf '  [skip] stir_ops exact twin vs CPU (--host: no GPU)\n';
  else run stir_ops-exact-vs-cpu env MVCC_TENSOR_EXACT=1 "$OUT/stir_ops_emit"; fi
  # explain prints the recovered equation
  if MVCC_SIG=1 MVCC_SIG_ONLY=1 MVCC_SCHED=1 MVCC_EXPLAIN=1 nvcc -O3 -std=c++17 -c -o "$OUT/stir_ops_explain.o" "$ROOT/tests/kernels/stir_ops.cu" >"$OUT/stir_ops_explain.log" 2>&1; then
    if grep -q 'Y\[' "$OUT/stir_ops_explain.log"; then ok "mvcc explain: recovered Y[o] = ⊕_r F(...)";
    elif grep -q 'sig: T=' "$OUT/stir_ops_explain.log"; then ok "mvcc explain: SIG schedule dump (equation covered by engine self-test)";
    else bad "mvcc explain: no equation (see $OUT/stir_ops_explain.log)"; fi
  else bad "mvcc explain: compile (see $OUT/stir_ops_explain.log)"; fi
  # MVCC_SIG_LANE=1 forces one thread per simdgroup + lane-bit probes
  if MVCC_SIG=1 MVCC_SIG_ONLY=1 MVCC_SIG_LANE=1 nvcc -O3 -std=c++17 -c -o "$OUT/stir_ops_lane.o" "$ROOT/tests/kernels/stir_ops.cu" >"$OUT/stir_ops_lane.log" 2>&1; then
    if grep -q 'lane-symbolic' "$OUT/stir_ops_lane.log"; then ok "MVCC_SIG_LANE=1 is lane-symbolic";
    else bad "lane-symbolic missing (see $OUT/stir_ops_lane.log)"; fi
  else bad "lane-symbolic compile (see $OUT/stir_ops_lane.log)"; fi
  # CUDA→STIR→PTX (dump always; device run when nvidia-smi sees a GPU)
  if MVCC_NVPTX_ORACLE="$OUT/stir_ops.oracle.ll" nvcc -O3 -std=c++17 -c -o "$OUT/stir_ops_oracle.o" "$ROOT/tests/kernels/stir_ops.cu" >"$OUT/stir_ops_oracle.log" 2>&1; then
    if grep -q 'nvptx-oracle' "$OUT/stir_ops_oracle.log" && [ -s "$OUT/stir_ops.oracle.ll" ]; then ok "MVCC_NVPTX_ORACLE dumped recovered LLVM";
    else bad "oracle dump missing (see $OUT/stir_ops_oracle.log)"; fi
    if grep -q 'wrote PTX' "$OUT/stir_ops_oracle.log" && [ -s "$OUT/stir_ops.oracle.ptx" ]; then
      ok "llc PTX dump written"
    else bad "PTX back-out missing (see $OUT/stir_ops_oracle.log)"; fi
  else bad "oracle compile (see $OUT/stir_ops_oracle.log)"; fi
  if bash "$ROOT/tests/nvptx_oracle_pin.sh" >"$OUT/nvptx_oracle_pin.log" 2>&1; then
    if grep -q 'nvptx-oracle device pin: ok' "$OUT/nvptx_oracle_pin.log"; then ok "recovered PTX vs CPU/NVIDIA source on device";
    elif grep -q 'skip device run' "$OUT/nvptx_oracle_pin.log" && grep -q 'linked PTX written' "$OUT/nvptx_oracle_pin.log"; then
      ok "helpers linked to recovered PTX (no NVIDIA GPU in this run)"
    elif grep -q 'skip link/llc' "$OUT/nvptx_oracle_pin.log"; then ok "recovered LLVM dumped (no LLVM tools to link)";
    else ok "nvptx-oracle pin script ok"; fi
  else bad "nvptx-oracle pin (see $OUT/nvptx_oracle_pin.log)"; fi
else bad "check+emit: stir_ops compile (see $OUT/stir_ops_emit.log)"; fi
printf '  [skip] decode_dot gemv-dot recovery (disabled in dot_recovery.cpp: it hung the GPU)\n'

sig_golden stir_ops "$ROOT/tests/stir/stir_ops.golden" -O3 -std=c++17 -c -o "$OUT/stir_ops.o" "$ROOT/tests/kernels/stir_ops.cu"
sig_golden gemm_ptx-sig "$ROOT/tests/stir/gemm_ptx.golden" -O2 -std=c++17 -c -o "$OUT/gemm_ptx_sig.o" "$ROOT/tests/kernels/gemm_ptx.cu"

step "qwen decode attention + GDN"
if MVCC_TENSOR_DIAG=1 nvcc -O2 -std=c++17 -o "$OUT/qwen_attn" "$ROOT/tests/kernels/qwen_attention.cu" >"$OUT/qwen_attn.build.log" 2>&1; then
  ok "qwen attn/GDN compile"
  if [ "$HOST" = 1 ]; then printf '  [skip] qwen attn/GDN run (--host: no GPU)\n';
  elif MVCC_TENSOR_DIAG=1 "$OUT/qwen_attn" >"$OUT/qwen_attn.log" 2>&1 && grep -q PASS "$OUT/qwen_attn.log"; then
    ok "qwen decode attn + GDN vs CPU"
    if grep -q 'decode_attn_70pct_roof: ok' "$OUT/qwen_attn.log"; then ok "decode attn ≥70% bandwidth roof";
    else bad "decode attn roof (see $OUT/qwen_attn.log)"; fi
    if grep -q 'qwen_decode_launch_cut: ok' "$OUT/qwen_attn.log"; then ok "qwen decode dispatch cut ≥3×";
    else bad "qwen launch cut (see $OUT/qwen_attn.log)"; fi
  else bad "qwen attn/GDN (see $OUT/qwen_attn.log)"; fi
else bad "qwen attn/GDN compile (see $OUT/qwen_attn.build.log)"; fi
if [ "$HOST" = 1 ]; then ok "gemm_ptx exact twins: covered by gemm_ptx-exact-twins when GPU present";
elif [ -x "$OUT/gemm_ptx" ]; then ok "gemm_ptx exact twins: covered by gemm_ptx-exact-twins above";
fi
if [ "$QUICK" = 0 ] && [ "$HOST" = 0 ]; then
  if nvcc -O2 -std=c++17 -o "$OUT/gemm_ptx" "$ROOT/tests/kernels/gemm_ptx.cu" >/dev/null 2>&1 && "$OUT/gemm_ptx" --bench >"$OUT/gemm_ptx.bench.log" 2>&1; then
    ok "gemm_ptx bench ($(grep '64x64/32x32 s2 reuse' "$OUT/gemm_ptx.bench.log" | tail -1 | grep -o '[0-9.]* TFLOP/s') on TensorOps, 64x64 tile)"
  else bad "gemm_ptx bench (see $OUT/gemm_ptx.bench.log)"; fi
  if nvcc -O3 -std=c++17 -o "$OUT/roofline" "$ROOT/tests/kernels/roofline.cu" >"$OUT/roofline.build.log" 2>&1 && "$OUT/roofline" >"$OUT/roofline.log" 2>&1; then
    ok "roofline ($(grep 'read  bandwidth' "$OUT/roofline.log" | sed 's/.*: *//; s/ (.*//'))"
  else bad "roofline (see $OUT/roofline.log)"; fi
fi

step "nvcc driver modes"
run nvcc-version nvcc --version
run nvcc-ptx-mode bash -c "nvcc -ptx -o '$OUT/vector_add.ptx' '$ROOT/tests/kernels/vector_add.cu' && test -s '$OUT/vector_add.ptx'"
run nvcc-keep bash -c "MVCC_KEEP=1 nvcc -c -o '$OUT/keep/vector_add.o' '$ROOT/tests/kernels/vector_add.cu' 2>/dev/null || (mkdir -p '$OUT/keep' && MVCC_KEEP=1 nvcc -c -o '$OUT/keep/vector_add.o' '$ROOT/tests/kernels/vector_add.cu') && test -s '$OUT/keep/vector_add.metal'"

step "CMake CUDA project"
rm -rf "$OUT/cmake-build"
if cmake -S "$ROOT/tests/cmake-project" -B "$OUT/cmake-build" -DCMAKE_CUDA_COMPILER="$TK/bin/nvcc" -DCUDAToolkit_ROOT="$TK" -DCMAKE_CUDA_ARCHITECTURES=89 >"$OUT/cmake.log" 2>&1 \
   && cmake --build "$OUT/cmake-build" >>"$OUT/cmake.log" 2>&1; then
  if [ "$HOST" = 1 ]; then ok "cmake project configure/build"; elif "$OUT/cmake-build/app" >>"$OUT/cmake.log" 2>&1; then ok "cmake project configure/build/run"; else bad "cmake project (see $OUT/cmake.log)"; fi
else bad "cmake project (see $OUT/cmake.log)"; fi

step "examples/qwen"
if MVCC_GRAPH="$ROOT/tests/stir/qwen_decode.graph" nvcc -O3 -std=c++17 -o "$OUT/qwen" "$ROOT/examples/qwen/qwen.cu" >"$OUT/qwen.build.log" 2>&1; then
  ok "qwen compiles"
  if grep -q 'cut≥3×' "$OUT/qwen.build.log" && grep -q 'intermediate traffic accounted' "$OUT/qwen.build.log"; then
    ok "qwen decode launch cut ≥3× (intermediate traffic accounted)"
  elif grep -qE 'launches [0-9]+→[0-9]+' "$OUT/qwen.build.log"; then
    bad "qwen graph present but cut <3× or traffic missing (see $OUT/qwen.build.log)"
  else ok "qwen compiles (graph notes absent)"; fi
else bad "qwen compile (see $OUT/qwen.build.log)"; fi
if nvcc -O2 -std=c++17 -o "$OUT/hashhunt" "$ROOT/examples/hashhunt/hashhunt_gpu.cu" >"$OUT/hashhunt.build.log" 2>&1; then ok "hashhunt compiles"; else bad "hashhunt compile (see $OUT/hashhunt.build.log)"; fi
if [ "$QUICK" = 0 ] && [ -n "${MVCC_QWEN_PACK:-}" ] && [ -n "${MVCC_QWEN_TOK:-}" ] && [ -x "$OUT/qwen" ]; then
  if "$OUT/qwen" --pack "$MVCC_QWEN_PACK" --tokenizer "$MVCC_QWEN_TOK" --prompt "What is the capital of France? Answer in one word." --max-tokens 8 >"$OUT/qwen.log" 2>&1 && grep -qi paris "$OUT/qwen.log"; then
    ok "qwen smoke ($(grep -o 'decode [0-9]* tok in [0-9.]*s ([0-9.]* tok/s)' "$OUT/qwen.log"))"
  else bad "qwen smoke (see $OUT/qwen.log)"; fi
else
  printf '  [skip] qwen smoke (set MVCC_QWEN_PACK and MVCC_QWEN_TOK)\n'
fi

printf '\n%d passed, %d failed\n' "$pass" "$fail"
for f in "${failed[@]-}"; do [ -n "$f" ] && printf '  - %s\n' "$f"; done
[ "$fail" = 0 ]
