#!/usr/bin/env bash
# Isolated diagnosis of M4 run 20261027-101109 against the PRESERVED image. No build, no host
# change; every container is --rm, --network none, and mounts are read-only.
#
#   IMAGE=bevserve:m4-20261027-101109 CKPT_DIR=... CKPT_SHA256=... \
#   HOST_STDLIB=~/anaconda3/envs/bevserve-m2/lib/python3.10 bash tools/m4_diagnose_lib2to3.sh
#
# Assertions (exit status 0 only if all hold):
#   A1  the preserved image cannot import lib2to3                      (reproduces failure 1)
#   A2  the preserved image cannot import distutils.core               (reproduces failure 2)
#   A3  the server's init path reports init_failed / model_stack_missing (explains startup-ready)
#   A4  with lib2to3 AND distutils from HOST_STDLIB mounted read-only on PYTHONPATH, the full
#       offline self-check passes: ok, identity equal, heatmap bit-identical, network unreachable
#       (shows nothing else is missing; the real fix installs Ubuntu's python3-lib2to3 and
#       python3-distutils instead of mounting host files)
set -uo pipefail
read -r -a DOCKER <<< "${DOCKER_CMD:-sudo docker}"
: "${IMAGE:?}" "${CKPT_DIR:?}" "${CKPT_SHA256:?}" "${HOST_STDLIB:?path to a Python 3.10 stdlib dir}"
[ -d "$HOST_STDLIB/lib2to3" ] && [ -f "$HOST_STDLIB/distutils/core.py" ] || { echo "HOST_STDLIB lacks lib2to3/distutils"; exit 2; }
REPO=$(cd "$(dirname "$0")/.." && pwd)
PUBLIC=$REPO/outputs/m4-20261027-101109/public-selfcheck
[ -f "$PUBLIC/selfcheck.json" ] || { echo "missing $PUBLIC/selfcheck.json"; exit 2; }
BASE=(run --rm --network none --label bevserve.m4.diagnose=1)
MODEL=(--gpus device=0 -v "$CKPT_DIR:/models:ro" -e "BEVSERVE_CHECKPOINT_SHA256=$CKPT_SHA256"
       -e "BEVSERVE_CHECKPOINT=/models/bevdet-r50-cbgs.pth")
MOUNTS=(-v "$HOST_STDLIB/lib2to3:/opt/diag/lib2to3:ro" -v "$HOST_STDLIB/distutils:/opt/diag/distutils:ro"
        -e PYTHONPATH=/opt/diag)
FAILED=0
assert() { if [ "$2" = yes ]; then echo "PASS $1"; else echo "FAIL $1"; FAILED=$((FAILED + 1)); fi; }
py() { "${DOCKER[@]}" "${BASE[@]}" "$@"; }

echo "== evidence: how distutils resolves in the preserved image (setuptools shim vs stdlib)"
py --entrypoint python "$IMAGE" -c "
import importlib, sys
for name in ('setuptools', 'setuptools._distutils', 'distutils.core'):
    try:
        m = importlib.import_module(name); print(f'{name}: OK {getattr(m, \"__file__\", \"\")}')
    except Exception as e:
        print(f'{name}: {type(e).__name__}: {e}')
" 2>&1 | tail -3

out=$(py --entrypoint python "$IMAGE" -c "import lib2to3" 2>&1 | tail -1); echo "  $out"
assert "A1 preserved image lacks lib2to3" "$(grep -q "No module named 'lib2to3'" <<< "$out" && echo yes || echo no)"
out=$(py --entrypoint python "$IMAGE" -c "import distutils.core" 2>&1 | tail -1); echo "  $out"
assert "A2 preserved image cannot import distutils.core" "$(grep -q -E "ModuleNotFoundError|ImportError" <<< "$out" && echo yes || echo no)"

out=$(py "${MODEL[@]}" --entrypoint python "$IMAGE" -m bevserve.selfcheck --readiness 2>/dev/null | tail -1); echo "  $out"
assert "A3 server init path: init_failed / model_stack_missing" \
  "$(grep -q '"status": "init_failed"' <<< "$out" && grep -q '"reason": "model_stack_missing"' <<< "$out" && echo yes || echo no)"

echo "== with lib2to3 + distutils mounted read-only (diagnostic only)"
py "${MOUNTS[@]}" --entrypoint python "$IMAGE" -c "import distutils.core, lib2to3; print('  distutils.core from', distutils.core.__file__)" 2>&1 | tail -1
out=$(py "${MODEL[@]}" "${MOUNTS[@]}" -v "$PUBLIC:/golden:ro" --entrypoint python "$IMAGE" \
        -m bevserve.selfcheck --golden /golden/selfcheck.json 2>/dev/null | tail -1)
echo "  $(python3 -c "import json,sys; r=json.loads(sys.argv[1]); print({k: r.get(k) for k in ('ok','init_error','golden','network_reachable')})" "$out" 2>&1)"
ok=$(python3 -c "
import json, sys
r = json.loads(sys.argv[1]); g = r.get('golden') or {}
print('yes' if r.get('ok') and g.get('identity_equal') and g.get('heatmap_bit_identical') and r.get('network_reachable') is False else 'no')
" "$out" 2>/dev/null || echo no)
assert "A4 full offline self-check passes with both modules present" "$ok"

if [ "$FAILED" -gt 0 ]; then echo "RESULT: $FAILED assertion(s) failed"; exit 1; fi
echo "RESULT: all assertions passed"
