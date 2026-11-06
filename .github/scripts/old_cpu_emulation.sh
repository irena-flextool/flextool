#!/usr/bin/env bash
# FlexTool on an emulated processor without AVX2 (qemu user-mode emulation).
#
#   old_cpu_emulation.sh <qemu-cpu-model> <venv> <flextool-source> <work-dir>
#
# <venv> is a virtualenv with FlexTool installed (non-editable) and only
# polars' default runtime.  Every Python process of the run, including the
# child processes FlexTool starts with sys.executable (the solver-stack
# probe, pip), runs under `qemu-x86_64 -cpu <model>`: <venv>/bin/python-emu
# is a wrapper that passes its own path as argv[0], so Python still finds
# the venv and sys.executable is the wrapper.  The qemu binary is $QEMU
# (default qemu-x86_64-static).  Checks, in order:
#   1. the emulated CPU hides AVX2 and polars' default runtime crashes on it
#      (otherwise the run would prove nothing);
#   2. `python -m flextool.env_check` names the missing features and the fix;
#   3. `import flextool` without a terminal stops with the pip command and
#      installs nothing;
#   4. `update_flextool.py --skip-git` (a copy of <flextool-source> in
#      <work-dir>) installs polars-runtime-compat of the polars version;
#   5. env_check passes and polars loads the compatible runtime;
#   6. the `base` example solves;
#   7. natively, on an AVX2 host, `import flextool` selects the default
#      runtime although the compatible one is installed.
set -euo pipefail

cpu=$1
venv=$(cd "$2" && pwd)
src=$(cd "$3" && pwd)
mkdir -p "$4"
work=$(cd "$4" && pwd)
qemu=${QEMU:-qemu-x86_64-static}
command -v "$qemu" >/dev/null || { echo "no $qemu" >&2; exit 2; }
ulimit -c 0  # no core files from the expected crash

real=$(readlink -f "$venv/bin/python")
emu="$venv/bin/python-emu"
cat > "$emu" <<EOF
#!/bin/sh
exec "$qemu" -cpu "$cpu" -0 "\$0" "$real" "\$@"
EOF
chmod +x "$emu"
export POLARS_MAX_THREADS=${POLARS_MAX_THREADS:-2}

step() { printf '\n=== %s\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }
run() {  # run "$@", print its output, keep the exit code in $rc and the output in $out
    set +e
    out=$("$@" 2>&1 </dev/null)
    rc=$?
    set -e
    printf '%s\n' "$out" | grep -v "TCG doesn't support requested feature" || true
    echo "(exit $rc)"
}
has_compat() { "$venv/bin/python" -m pip show polars-runtime-compat >/dev/null 2>&1; }

step "1. $cpu: no AVX2 in cpuid, and polars' default runtime crashes"
has_compat && fail "the venv must start with polars' default runtime only"
# polars' stdlib-only checker, loaded from its file: importing polars would load the runtime.
run "$emu" -c "
import importlib.util as u, os, sys
d = u.find_spec('polars').submodule_search_locations[0]
s = u.spec_from_file_location('cpu_check', os.path.join(d, '_cpu_check.py'))
m = u.module_from_spec(s)
s.loader.exec_module(m)
f = m._read_cpu_flags()
print('missing:', [k for k, v in f.items() if not v])
sys.exit(1 if f['avx2'] else 0)"
[ "$rc" -eq 0 ] || fail "the emulated $cpu reports AVX2; the run would prove nothing"
run env POLARS_FORCE_PKG=32 FLEXTOOL_NO_ENV_CHECK=1 "$emu" -c \
    "import polars as pl; print(pl.DataFrame({'a': list(range(1000))}).sort('a', descending=True)['a'][0])"
[ "$rc" -ne 0 ] || fail "polars' default runtime ran on the emulated $cpu; the run would prove nothing"

step "2. env_check names the missing features and the fix"
run "$emu" -m flextool.env_check
[ "$rc" -eq 1 ] || fail "env_check exited $rc, expected 1"
grep -q "needs the compatible polars build" <<<"$out" || fail "env_check did not say the compatible build is needed"
grep -q 'polars\[rtcompat\]==' <<<"$out" || fail "env_check did not print the pip command"

step "3. import flextool without a terminal stops and installs nothing"
run "$emu" -c "import flextool"
[ "$rc" -ne 0 ] || fail "import flextool succeeded with the default runtime only"
grep -q "CompatRuntimeRequired" <<<"$out" || fail "import flextool did not stop with CompatRuntimeRequired"
has_compat && fail "the import installed the compatible runtime without a yes"

step "4. update_flextool.py --skip-git installs the compatible runtime"
rm -rf "$work/flextool"
cp -r "$src" "$work/flextool"
(cd "$work/flextool" && "$emu" update_flextool.py --skip-git 2>&1 | grep -v "TCG doesn't support requested feature")
"$venv/bin/python" -c "
from importlib.metadata import version as v
p, c = v('polars'), v('polars-runtime-compat')
print('polars', p, 'polars-runtime-compat', c)
assert p == c, (p, c)" || fail "polars-runtime-compat missing or of another version"

step "5. env_check passes; polars loads the compatible runtime"
run "$emu" -m flextool.env_check
[ "$rc" -eq 0 ] || fail "env_check exited $rc after the fix"
grep -q "loaded: rtcompat" <<<"$out" || fail "polars did not load the compatible runtime"

step "6. the base example solves"
(
    cd "$work/flextool"
    rm -rf output_parquet/base
    "$emu" -m flextool.cli.cmd_run_flextool sqlite:///templates/examples.sqlite --scenario-name base \
        --write-methods parquet 2>&1 | grep -v "TCG doesn't support requested feature" | tail -20
    n=$(find output_parquet/base -name '*.parquet' | wc -l)
    echo "$n parquet files in output_parquet/base"
    [ "$n" -gt 0 ] || fail "no parquet output"
)

step "7. natively: the default runtime is selected on an AVX2 host"
if grep -qw avx2 /proc/cpuinfo; then
    run "$venv/bin/python" -c "import os, flextool, polars._plr as p; print(p.RUNTIME_REPR, os.environ.get('POLARS_PREFER_PKG')); assert p.RUNTIME_REPR in ('rt32', 'rt64')"
    [ "$rc" -eq 0 ] || fail "import flextool did not select the default runtime on this AVX2 host"
else
    echo "host has no AVX2: skipped"
fi
printf '\nOK: %s\n' "$cpu"
