#!/usr/bin/env bash
# Targeted M4 retest: real-frame HTTP parity of an ALREADY BUILT image against the M3 golden.
# No build. Uses `sudo docker` by default. Starts one Compose project unique to this run, bound to
# 127.0.0.1, and removes only that project on exit (logs saved first). The nuScenes-derived golden
# is read on the host only.
#
#   IMAGE=bevserve:m4-20261027-112843 BEVDET_ROOT=... CKPT_DIR=... CKPT_SHA256=... \
#   GOLDEN_DIR=... HOST_PY=... bash tools/m4_parity_retest.sh
#
# Exit 0 only if every golden frame returns HTTP 200 with byte-identical boxes, the same model
# identity, frame and units.
set -uo pipefail
umask 077
read -r -a DOCKER <<< "${DOCKER_CMD:-sudo docker}"
d() { "${DOCKER[@]}" "$@"; }
REPO=$(cd "$(dirname "$0")/.." && pwd)
RUN_ID=$(date +%Y%m%d-%H%M%S)
PROJECT=bevserve-m4-parity-$RUN_ID
PORT=${PORT:-8000}
GPU=${GPU:-0}
CKPT_FILE=${CKPT_FILE:-bevdet-r50-cbgs.pth}
OUT=$REPO/outputs/m4-parity-$RUN_ID
mkdir -p "$OUT" && chmod 700 "$OUT"
COMPOSE_USED=0
fail() { echo "FAIL $*"; exit 1; }

for var in IMAGE BEVDET_ROOT CKPT_DIR CKPT_SHA256 GOLDEN_DIR HOST_PY; do [ -n "${!var:-}" ] || fail "$var not set"; done
for var in BEVDET_ROOT CKPT_DIR GOLDEN_DIR; do
  abs=$(cd "${!var}" 2>/dev/null && pwd) || fail "$var is not a directory"
  printf -v "$var" '%s' "$abs"
done
case "$HOST_PY" in /*) ;; *) HOST_PY=$(realpath -s "$HOST_PY") ;; esac
[ -r "$GOLDEN_DIR/http_golden.json" ] || fail "golden missing: $GOLDEN_DIR/http_golden.json"
[ "$(sha256sum "$CKPT_DIR/$CKPT_FILE" | cut -d' ' -f1)" = "$CKPT_SHA256" ] || fail "checkpoint sha256 mismatch"
python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['model']['checkpoint_sha256']==sys.argv[2] else 1)" \
  "$GOLDEN_DIR/http_golden.json" "$CKPT_SHA256" || fail "golden was not recorded with this checkpoint"

cleanup() {
  local code=$?
  trap - EXIT INT TERM
  if [ "$COMPOSE_USED" = 1 ]; then
    d compose -p "$PROJECT" logs --no-color > "$OUT/compose-logs.txt" 2>&1 || true
    d compose -p "$PROJECT" down --remove-orphans > /dev/null 2>&1 || true
  fi
  echo "results: $OUT"; exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

d image inspect -f 'id={{.Id}} size_bytes={{.Size}} created={{.Created}}' "$IMAGE" > "$OUT/image.txt" 2>/dev/null \
  || fail "image $IMAGE not found"
echo "image: $(cat "$OUT/image.txt")"
ss -ltnH "( sport = :$PORT )" | grep -q . && fail "port $PORT already in use"
cat > "$OUT/compose.env" <<EOF
BEVSERVE_IMAGE=$IMAGE
BEVSERVE_CHECKPOINT_DIR=$CKPT_DIR
BEVSERVE_CHECKPOINT_FILE=$CKPT_FILE
BEVSERVE_CHECKPOINT_SHA256=$CKPT_SHA256
BEVSERVE_HOST_PORT=$PORT
BEVSERVE_GPU=$GPU
EOF
COMPOSE_USED=1
d compose -p "$PROJECT" -f "$REPO/docker/compose.yaml" --env-file "$OUT/compose.env" up -d --no-build > "$OUT/compose-up.txt" 2>&1 \
  || fail "compose up"
for _ in $(seq 1 240); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:$PORT/readyz")" = 200 ] && break; sleep 1
done
[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/readyz")" = 200 ] || fail "service not ready"

( cd "$BEVDET_ROOT" && PYTHONDONTWRITEBYTECODE=1 "$HOST_PY" "$REPO/tools/golden_http.py" compare \
    --url "http://127.0.0.1:$PORT" --golden "$GOLDEN_DIR/http_golden.json" ) \
  > "$OUT/golden_compare.json" 2> "$OUT/golden_compare.stderr"
python3 - "$OUT/golden_compare.json" <<'PY' || exit 1
import json, sys
r = json.load(open(sys.argv[1]))
for f in r["frames"]:
    print(f"frame {f['index']:5d}: status {f['status']} boxes {f.get('count')} "
          f"identical={f.get('boxes_identical')} frame/units={f.get('frame_units_ok')} identity={f.get('identity_ok')}")
print("PASS http-golden-parity" if r["pass"] else "FAIL http-golden-parity")
sys.exit(0 if r["pass"] else 1)
PY
