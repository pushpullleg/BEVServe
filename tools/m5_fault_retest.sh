#!/usr/bin/env bash
# Targeted retest of the M4 fault-recovery check on an existing, verified image (no build).
#
#   IMAGE=bevserve:<tag> EXPECT_IMAGE_ID=sha256:<id> CKPT_DIR=... CKPT_SHA256=... HOST_PY=... \
#   bash tools/m5_fault_retest.sh
#
# Checks: image id and source provenance (bevserve package in the image == this work tree), then
# fault injection (call 2 is fatal): call 1 -> 200, call 2 -> 500 inference_fatal; the Docker die
# event of that container carries exitCode 70 (daemon evidence, not logs or restart count); Docker
# restarts it exactly once; ready again; a request after recovery -> 200; no further restarts.
#
# Safety: `sudo docker` only for this run's own Compose project bevserve-m5fault-<run id>; cleanup
# (logs saved first, then down with the same -p/-f/--env-file) on exit and Ctrl-C; no build, prune,
# global kill or remove; 127.0.0.1 only; synthetic inputs; private outputs in
# outputs/m5-fault-<run id>/ (umask 077, 0700).
set -uo pipefail
umask 077

REPO=$(cd "$(dirname "$0")/.." && pwd)
RUN_ID=$(date +%Y%m%d-%H%M%S)
PROJECT=bevserve-m5fault-$RUN_ID
PORT=${PORT:-8000}
GPU=${GPU:-0}
CKPT_FILE=${CKPT_FILE:-bevdet-r50-cbgs.pth}
read -r -a DOCKER <<< "${DOCKER_CMD:-sudo docker}"
d() { "${DOCKER[@]}" "$@"; }
OUT=$REPO/outputs/m5-fault-$RUN_ID
mkdir -p "$OUT" && chmod 700 "$OUT"
LOG=$OUT/checks.log SUMMARY=$OUT/summary.txt ENVF=$OUT/compose.env OVERRIDE=$OUT/compose.fault.yaml
: > "$LOG"; : > "$SUMMARY"
exec 3>&1 4>&2
DC=(compose -p "$PROJECT" -f "$REPO/docker/compose.yaml" -f "$OVERRIDE" --env-file "$ENVF")
FAILED=0
UP=0
SUDO_KEEPALIVE=

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
pass() { echo "PASS $*" | tee -a "$SUMMARY" >> "$LOG"; echo "PASS $*"; }
fail() { echo "FAIL $*" | tee -a "$SUMMARY" >> "$LOG"; echo "FAIL $*"; FAILED=$((FAILED + 1)); }
die() { fail "$1 (critical; stopping)"; exit 1; }
http_code() { curl -s -o /dev/null -w '%{http_code}' --max-time 3 "$@"; }
wait_ready() {  # prints seconds until /readyz is 200; non-zero on timeout
  local t0=$SECONDS
  while (( SECONDS - t0 < $1 )); do
    [ "$(http_code "http://127.0.0.1:$PORT/readyz")" = 200 ] && { echo $((SECONDS - t0)); return 0; }
    sleep 1
  done
  return 1
}

cleanup() {
  local code=$?
  trap - EXIT INT TERM
  exec 1>&3 2>&4
  if [ "$UP" = 1 ]; then   # only this run's project, with the same files and env file as `up`
    log "cleanup: saving logs, then removing Compose project $PROJECT"
    d "${DC[@]}" logs --no-color --timestamps > "$OUT/compose-logs.txt" 2>&1 || true
    d "${DC[@]}" down --remove-orphans >> "$LOG" 2>&1 || true
    left=$(d ps -aq --filter "label=com.docker.compose.project=$PROJECT")
    [ -z "$left" ] || { echo "FAIL cleanup: containers of $PROJECT remain: $left" | tee -a "$SUMMARY"; FAILED=$((FAILED + 1)); }
  fi
  [ -n "$SUDO_KEEPALIVE" ] && kill "$SUDO_KEEPALIVE" 2>/dev/null
  if [ "$code" -ne 0 ] && [ "$FAILED" -eq 0 ]; then FAILED=1; echo "FAIL run aborted (exit $code)" >> "$SUMMARY"; fi
  echo "== summary ($OUT/summary.txt)"; cat "$SUMMARY"
  if [ "$FAILED" -gt 0 ]; then echo "RESULT: $FAILED check(s) failed"; exit 1; fi
  echo "RESULT: all checks passed"; exit 0
}
trap cleanup EXIT
trap 'exec 1>&3 2>&4; log "interrupted"; exit 130' INT TERM

# ------------------------------------------------------------------- inputs and prerequisites
log "== inputs and prerequisites (run $RUN_ID, project $PROJECT, image ${IMAGE:-unset})"
for var in IMAGE CKPT_DIR CKPT_SHA256 HOST_PY; do [ -n "${!var:-}" ] || die "input: $var is not set"; done
CKPT_DIR=$(cd "$CKPT_DIR" 2>/dev/null && pwd) || die "input: CKPT_DIR is not a directory"
case "$HOST_PY" in /*) ;; *) HOST_PY=$(realpath -s "$HOST_PY") ;; esac
"$HOST_PY" -c "import httpx" > /dev/null 2>&1 || die "input: HOST_PY lacks httpx"
[ "$(sha256sum "$CKPT_DIR/$CKPT_FILE" | cut -d' ' -f1)" = "$CKPT_SHA256" ] || die "input: checkpoint sha256 mismatch"
if [[ "${DOCKER[0]}" == sudo ]]; then
  sudo -v || die "prerequisite: sudo authentication"
  ( while sleep 60; do sudo -n -v 2>/dev/null || exit; done ) & SUDO_KEEPALIVE=$!
fi
IMAGE_ID=$(d image inspect -f '{{.Id}}' "$IMAGE" 2>> "$LOG") || die "prerequisite: image $IMAGE not found"
echo "image $IMAGE id $IMAGE_ID" >> "$LOG"
if [ -n "${EXPECT_IMAGE_ID:-}" ]; then
  [ "$IMAGE_ID" = "$EXPECT_IMAGE_ID" ] && pass "image id is the verified $IMAGE_ID" || die "image id $IMAGE_ID != EXPECT_IMAGE_ID"
fi
if ss -ltnH "( sport = :$PORT )" | grep -q .; then die "prerequisite: port $PORT already in use"; fi
[ -z "$(d ps -aq --filter "label=com.docker.compose.project=$PROJECT")" ] || die "prerequisite: project $PROJECT exists"
BUSY=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' '$2 > 1024')
[ -z "$BUSY" ] || die "prerequisite: another GPU workload is running ($BUSY)"

# ------------------------------------------------------------------- provenance
log "== provenance"
python3 "$REPO/tools/provenance.py" source --repo "$REPO" > "$OUT/provenance-source.json" || die "source provenance"
d run --rm -i --network none --entrypoint python "$IMAGE" - package < "$REPO/tools/provenance.py" \
  > "$OUT/provenance-image.json" 2>> "$LOG" || die "image package provenance"
if python3 "$REPO/tools/provenance.py" compare "$OUT/provenance-source.json" "$OUT/provenance-image.json" \
     > "$OUT/provenance-compare.json"; then
  pass "bevserve package in the image is byte-identical to the work tree's src/bevserve"
else
  fail "bevserve package in the image differs from the work tree (provenance-compare.json)"
fi
GIT_DESCRIBE=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['git_describe'])" "$OUT/provenance-source.json")
log "source: $GIT_DESCRIBE (details: provenance-source.json)"

# ------------------------------------------------------------------- fault injection
log "== fatal runtime error -> exit 70 -> supervised restart -> ready"
cat > "$ENVF" <<EOF
BEVSERVE_IMAGE=$IMAGE
BEVSERVE_CHECKPOINT_DIR=$CKPT_DIR
BEVSERVE_CHECKPOINT_FILE=$CKPT_FILE
BEVSERVE_CHECKPOINT_SHA256=$CKPT_SHA256
BEVSERVE_HOST_PORT=$PORT
BEVSERVE_GPU=$GPU
BEVSERVE_GIT_COMMIT=$GIT_DESCRIBE
EOF
cat > "$OVERRIDE" <<'EOF'
services:
  bevserve:
    entrypoint: ["python", "-m", "bevserve.faultinject", "--fatal-on-call", "2"]
    environment:
      BEVSERVE_FAULT_INJECTION_ACK: "1"
EOF
UP=1
d "${DC[@]}" up -d --no-build >> "$LOG" 2>&1 || die "compose up"
FID=$(d "${DC[@]}" ps -q bevserve) && [ -n "$FID" ] && FID=$(d inspect -f '{{.Id}}' "$FID") || die "container id"
secs=$(wait_ready 240) && pass "startup-ready (${secs} s)" || die "startup-ready"
POLICY=$(d inspect -f '{{.HostConfig.RestartPolicy.Name}}:{{.HostConfig.RestartPolicy.MaximumRetryCount}}' "$FID")
[ "$POLICY" = "on-failure:5" ] && pass "restart-policy ($POLICY)" || fail "restart-policy ($POLICY)"

RC0=$(d inspect -f '{{.RestartCount}}' "$FID")
T_START=$(date +%s.%N)   # fractional --since; the window ends in the future (see below)
fault_requests() {  # fault_requests <count>: synthetic requests; prints "<status>:<type>" per call
  PYTHONPATH="$REPO" PYTHONDONTWRITEBYTECODE=1 "$HOST_PY" - "$PORT" "$1" 2>> "$LOG" <<'PY'
import sys, time, httpx
from tests.synthetic import multipart_files
port, n, seen = sys.argv[1], int(sys.argv[2]), []
files, data = multipart_files()
for _ in range(n):
    try:
        r = httpx.post(f"http://127.0.0.1:{port}/v1/detect", files=files, data=data, timeout=30)
        seen.append(f"{r.status_code}:{r.json().get('type', 'ok')}")
    except httpx.HTTPError as exc:
        seen.append(type(exc).__name__)
    time.sleep(0.2)
print(" ".join(seen))
PY
}
fault_requests 6 > "$OUT/fault_requests.txt"
cat "$OUT/fault_requests.txt" >> "$LOG"
grep -q -E '^200:ok 500:/errors/inference_fatal' "$OUT/fault_requests.txt" \
  && pass "call 1 -> 200, call 2 -> 500 inference_fatal" || fail "call 1 -> 200, call 2 -> 500 inference_fatal"
t0=$SECONDS
while (( SECONDS - t0 < 120 )) && [ "$(d inspect -f '{{.RestartCount}}' "$FID")" = "$RC0" ]; do sleep 1; done
RC1=$(d inspect -f '{{.RestartCount}}' "$FID")
# Docker drops events later than --until; ending the window 2 s in the future includes the current
# second (the old whole-second "now" could cut off the die event). The command returns at --until.
T_END=$(( $(date +%s) + 2 ))
echo "events window: --since $T_START --until $T_END; restart count $RC0 -> $RC1" >> "$LOG"
d events --since "$T_START" --until "$T_END" --filter type=container \
  --filter "label=com.docker.compose.project=$PROJECT" --format '{{json .}}' > "$OUT/fault_events.jsonl" 2>> "$LOG" \
  || fail "docker events query"
if python3 "$REPO/tools/fault_events.py" --events "$OUT/fault_events.jsonl" --container "$FID" --exit-code 70 \
     --restarts-before "$RC0" --restarts-after "$RC1" > "$OUT/fault_events.json" 2>> "$LOG"; then
  pass "process exited with code 70 (docker die event, then restart; restart count $RC0 -> $RC1)"
else
  fail "process exited with code 70 (see fault_events.json)"
fi
cat "$OUT/fault_events.json" >> "$LOG"
d logs "$FID" 2>&1 | grep -q "exiting with code 70" && pass "fatal exit logged by the service" || fail "fatal exit logged by the service"
secs=$(wait_ready 240) && pass "ready again after restart (${secs} s)" || fail "ready again after restart"
fault_requests 1 > "$OUT/recovery_request.txt"
grep -q -x '200:ok' "$OUT/recovery_request.txt" && pass "request after recovery -> 200" \
  || fail "request after recovery -> 200 (saw: $(cat "$OUT/recovery_request.txt"))"
sleep 10
RC2=$(d inspect -f '{{.RestartCount}}' "$FID")
[ "$RC2" = "$RC1" ] && pass "no further restarts (restart count $RC2 after 10 s)" || fail "further restarts ($RC1 -> $RC2)"
log "done"
