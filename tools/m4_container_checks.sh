#!/usr/bin/env bash
# M4 container verification. Run by the operator after the host setup in docs/host-setup.md.
#
#   BEVDET_ROOT=/path/to/BEVDet CKPT_DIR=/path/to/checkpoint/dir CKPT_SHA256=<sha> \
#   GOLDEN_DIR=/path/to/golden HOST_PY=/path/to/model-env/bin/python \
#   bash tools/m4_container_checks.sh
#
# Safety properties:
# - Uses `sudo docker` by default (no docker-group membership). Never runs prune, global kill or
#   global remove: cleanup removes only the containers, Compose project and build-stage tag that
#   THIS run created (unique run id, recorded container ids, project bevserve-m4-<run id>).
# - Every check is recorded as PASS/FAIL; critical failures stop the run; the exit status is
#   non-zero if any check failed or the run was interrupted. Cleanup runs on exit and on Ctrl-C.
# - Results are private (umask 077, outputs/m4-<run id>/ is 0700). The nuScenes-derived HTTP golden
#   is read on the host only and is never copied into or mounted into a container. Only the
#   synthetic self-check golden is mounted (read-only, world-readable copy) for the non-root user.
# - All inputs and prerequisites are verified before anything is built.
#
# GOLDEN_DIR holds selfcheck.json (python -m bevserve.selfcheck --record) and http_golden.json
# (tools/golden_http.py record), both made with the M3-validated host environment.
set -uo pipefail
umask 077

REPO=$(cd "$(dirname "$0")/.." && pwd)
RUN_ID=$(date +%Y%m%d-%H%M%S)
PROJECT=bevserve-m4-$RUN_ID
IMAGE=${IMAGE:-bevserve:m4-$RUN_ID}
BUILD_TAG=$IMAGE-buildstage
PORT=${PORT:-8000}
GPU=${GPU:-0}
CKPT_FILE=${CKPT_FILE:-bevdet-r50-cbgs.pth}
read -r -a DOCKER <<< "${DOCKER_CMD:-sudo docker}"
OUT=$REPO/outputs/m4-$RUN_ID
LABEL=bevserve.m4.run=$RUN_ID
mkdir -p "$OUT" && chmod 700 "$OUT"
LOG=$OUT/checks.log SUMMARY=$OUT/summary.txt
: > "$LOG"; : > "$SUMMARY"
# Keep the terminal's stdout/stderr: a trap can fire while a command's output is redirected
# (e.g. during `docker build > build.log`), and its report must still reach the operator.
exec 3>&1 4>&2

FAILED=0
CREATED=()          # container ids started by this run (docker run)
COMPOSE_USED=0      # this run created Compose project $PROJECT
DC=()               # compose CLI args (-p, -f files, --env-file) of the active configuration
BUILD_TAG_MADE=0
SUDO_KEEPALIVE=

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
pass() { echo "PASS $*" | tee -a "$SUMMARY" >> "$LOG"; echo "PASS $*"; }
fail() { echo "FAIL $*" | tee -a "$SUMMARY" >> "$LOG"; echo "FAIL $*"; FAILED=$((FAILED + 1)); }
check() {  # check <name> <command...>: PASS iff the command exits 0
  local name=$1; shift
  if "$@" >> "$LOG" 2>&1; then pass "$name"; else fail "$name"; fi
}
die() { fail "$1 (critical; stopping)"; exit 1; }
d() { "${DOCKER[@]}" "$@"; }

# ------------------------------------------------------------------- cleanup (this run only)
cleanup() {
  local code=$?
  trap - EXIT INT TERM
  exec 1>&3 2>&4
  log "cleanup: removing only resources created by run $RUN_ID"
  for cid in "${CREATED[@]}"; do d rm -f "$cid" >> "$LOG" 2>&1 || true; done
  if [ "$COMPOSE_USED" = 1 ]; then   # same -p/-f/--env-file as `up` (the file needs the env file)
    d "${DC[@]}" logs --no-color > "$OUT/compose-logs.txt" 2>&1 || true   # evidence first
    d "${DC[@]}" down --remove-orphans >> "$LOG" 2>&1 || true
  fi
  if [ "$BUILD_TAG_MADE" = 1 ]; then d image rm "$BUILD_TAG" >> "$LOG" 2>&1 || true; fi
  [ -n "$SUDO_KEEPALIVE" ] && kill "$SUDO_KEEPALIVE" 2>/dev/null
  if [ "$code" -ne 0 ] && [ "$FAILED" -eq 0 ]; then FAILED=1; echo "FAIL run aborted (exit $code)" >> "$SUMMARY"; fi
  echo "== summary ($OUT/summary.txt)"; cat "$SUMMARY"
  echo "image kept for review: $IMAGE"
  if [ "$FAILED" -gt 0 ]; then echo "RESULT: $FAILED check(s) failed"; exit 1; fi
  echo "RESULT: all checks passed"; exit 0
}
trap cleanup EXIT
trap 'exec 1>&3 2>&4; log "interrupted"; exit 130' INT TERM

LAST_CID=
run_labeled() {  # run_labeled <docker run args...>: detached; id in LAST_CID and recorded for cleanup
  LAST_CID=$(d run -d --label "$LABEL" "$@") || return 1
  CREATED+=("$LAST_CID")
}
remove_container() {
  d rm -f "$1" >> "$LOG" 2>&1
  local keep=() c
  for c in "${CREATED[@]}"; do [ "$c" != "$1" ] && keep+=("$c"); done
  CREATED=("${keep[@]+"${keep[@]}"}")
}
http_code() { curl -s -o /dev/null -w '%{http_code}' "$@"; }
readyz() { curl -s --max-time 3 "http://127.0.0.1:$PORT/readyz"; }
wait_ready() {  # prints seconds until /readyz is 200; non-zero on timeout
  local t0=$SECONDS
  while (( SECONDS - t0 < $1 )); do
    [ "$(http_code --max-time 3 "http://127.0.0.1:$PORT/readyz")" = 200 ] && { echo $((SECONDS - t0)); return 0; }
    sleep 1
  done
  return 1
}
json_ok() {  # json_ok <python expr over d, sys.argv> <json file> [extra args]
  local expr=$1; shift
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if ($expr) else 1)" "$@"
}

# ------------------------------------------------------------------- 0. inputs and prerequisites
log "== 0. inputs and prerequisites (run $RUN_ID, project $PROJECT, image $IMAGE)"
for var in BEVDET_ROOT CKPT_DIR CKPT_SHA256 GOLDEN_DIR HOST_PY; do
  [ -n "${!var:-}" ] || die "input: $var is not set"
done
# Make every path absolute now: later steps cd into BEVDET_ROOT (run 20261027-112843 failed its
# parity step because a relative GOLDEN_DIR no longer resolved after that cd).
for var in BEVDET_ROOT CKPT_DIR GOLDEN_DIR; do
  abs=$(cd "${!var}" 2>/dev/null && pwd) || die "input: $var is not a directory"
  printf -v "$var" '%s' "$abs"
done
case "$HOST_PY" in /*) ;; *) HOST_PY=$(realpath -s "$HOST_PY") ;; esac   # keep env symlinks
[[ "${PARALLEL_THREADS:-12}" =~ ^[0-9]+$ ]] && (( ${PARALLEL_THREADS:-12} >= 1 && ${PARALLEL_THREADS:-12} <= 32 )) \
  || die "input: PARALLEL_THREADS must be an integer in [1, 32]"
for tool in curl ss python3 sha256sum; do command -v "$tool" > /dev/null || die "input: $tool not found"; done
[ -d "$BEVDET_ROOT/mmdet3d" ] || die "input: BEVDET_ROOT is not a BEVDet checkout"
[ -r "$BEVDET_ROOT/data/nuscenes/bevdetv3-trainval_infos_val.pkl" ] || die "input: val info file missing"
[ -d "$CKPT_DIR" ] && [ -r "$CKPT_DIR/$CKPT_FILE" ] || die "input: checkpoint $CKPT_FILE not readable in CKPT_DIR"
[ -r "$GOLDEN_DIR/selfcheck.json" ] && [ -r "$GOLDEN_DIR/http_golden.json" ] || die "input: golden files missing"
[ -x "$HOST_PY" ] || die "input: HOST_PY not executable"
"$HOST_PY" -c "import httpx, mmcv, numpy" > /dev/null 2>&1 || die "input: HOST_PY lacks httpx/mmcv/numpy"
ACTUAL_SHA=$(sha256sum "$CKPT_DIR/$CKPT_FILE" | cut -d' ' -f1)
[ "$ACTUAL_SHA" = "$CKPT_SHA256" ] || die "input: checkpoint sha256 differs from CKPT_SHA256"
CKPT_SIZE=$(stat -c %s "$CKPT_DIR/$CKPT_FILE")
for g in selfcheck.json http_golden.json; do
  json_ok "d['model']['checkpoint_sha256'] == sys.argv[2]" "$GOLDEN_DIR/$g" "$CKPT_SHA256" 2>> "$LOG" \
    || die "input: $g was not recorded with this checkpoint"
done
chmod 700 "$GOLDEN_DIR" && chmod 600 "$GOLDEN_DIR/http_golden.json" || die "input: cannot make the HTTP golden private"
pass "inputs verified (checkpoint sha256, goldens, host client)"

if [[ "${DOCKER[0]}" == sudo ]]; then
  sudo -v || die "prerequisite: sudo authentication"
  ( while sleep 60; do sudo -n -v 2>/dev/null || exit; done ) & SUDO_KEEPALIVE=$!
fi
d version --format '{{.Server.Version}}' >> "$LOG" 2>&1 || die "prerequisite: docker daemon not reachable"
d compose version >> "$LOG" 2>&1 || die "prerequisite: docker compose plugin missing"
d buildx version >> "$LOG" 2>&1 || die "prerequisite: docker buildx missing"
d info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia || die "prerequisite: nvidia runtime not registered"
[ "$(uname -r)" = "6.14.0-37-generic" ] || die "prerequisite: kernel is not the pinned 6.14.0-37-generic"
nvidia-smi --query-gpu=driver_version,name --format=csv,noheader >> "$LOG"
if ss -ltnH "( sport = :$PORT )" | grep -q .; then die "prerequisite: port $PORT already in use"; fi
if [ -n "$(d ps -aq --filter "label=com.docker.compose.project=$PROJECT")" ]; then die "prerequisite: project $PROJECT exists"; fi
if d image inspect "$IMAGE" > /dev/null 2>&1; then die "prerequisite: image tag $IMAGE already exists"; fi
BUSY=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' '$2 > 1024')
[ -z "$BUSY" ] || die "prerequisite: another GPU workload is running ($BUSY)"
echo "unrelated containers present (untouched): $(d ps -q | wc -l)" >> "$LOG"
check "gpu-in-container" d run --rm --label "$LABEL" --gpus "device=$GPU" nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi -L
PUBLIC=$OUT/public-selfcheck   # synthetic golden only; readable by the container's non-root user
mkdir -p "$PUBLIC" && cp "$GOLDEN_DIR/selfcheck.json" "$PUBLIC/" && chmod 755 "$PUBLIC" && chmod 644 "$PUBLIC/selfcheck.json"

# ------------------------------------------------------------------- 1. build and provenance
log "== 1. build"
t0=$SECONDS
DOCKER_BUILDKIT=1 d build -f "$REPO/docker/Dockerfile.gpu" -t "$IMAGE" "$REPO" > "$OUT/build.log" 2>&1 \
  || die "build (see build.log)"
pass "build ($((SECONDS - t0)) s)"
DOCKER_BUILDKIT=1 d build --target build -f "$REPO/docker/Dockerfile.gpu" -t "$BUILD_TAG" "$REPO" >> "$OUT/build.log" 2>&1 \
  && BUILD_TAG_MADE=1 || die "build-stage tag for scanning"
d image inspect -f 'id={{.Id}} size_bytes={{.Size}} created={{.Created}}' "$IMAGE" | tee -a "$LOG" > "$OUT/image.txt"
d history --no-trunc --format '{{.Size}}\t{{.CreatedBy}}' "$IMAGE" > "$OUT/history.txt"
d run --rm --label "$LABEL" --network none --entrypoint cat "$IMAGE" /opt/venv/bevserve-manifest.txt > "$OUT/manifest.txt" \
  && pass "dependency manifest recorded" || fail "dependency manifest recorded"
d run --rm --label "$LABEL" --network none --entrypoint cat "$IMAGE" /opt/venv/bevserve-pip-check.txt > "$OUT/pip-check.txt" || true
nonroot() { d run --rm --label "$LABEL" --network none --entrypoint id "$IMAGE" | grep -q 'uid=10001'; }
check "runs-as-non-root" nonroot

scan_image() {  # scan_image <tag> <name>: every layer, via docker save; PASS only on a clean full read
  local out=$OUT/scan-$2.json
  if d save "$1" | python3 "$REPO/tools/scan_image_layers.py" --checkpoint-size "$CKPT_SIZE" \
       --checkpoint-sha256 "$CKPT_SHA256" > "$out" 2>> "$LOG" && json_ok "d['ok'] and d['layers'] > 0" "$out"; then
    pass "layer-scan $2 ($(python3 -c "import json;d=json.load(open('$out'));print(d['layers'],'layers',d['files'],'files')"))"
  else
    fail "layer-scan $2 (see scan-$2.json)"
  fi
}
scan_image "$IMAGE" runtime-image
scan_image "$BUILD_TAG" build-stage
d image rm "$BUILD_TAG" >> "$LOG" 2>&1 && BUILD_TAG_MADE=0

# ------------------------------------------------------------------- 2. offline verification
log "== 2. offline verification (--network none; not the serving setup)"
d run --rm --label "$LABEL" --network none --gpus "device=$GPU" -v "$CKPT_DIR:/models:ro" -v "$PUBLIC:/golden:ro" \
  -e "BEVSERVE_CHECKPOINT_SHA256=$CKPT_SHA256" -e "BEVSERVE_CHECKPOINT=/models/$CKPT_FILE" \
  --entrypoint python "$IMAGE" -m bevserve.selfcheck --golden /golden/selfcheck.json > "$OUT/selfcheck.json" 2> "$OUT/selfcheck.stderr"
check "offline-selfcheck (ops load, sha256, heatmap bit-identical, no network)" \
  json_ok "d['ok'] and d['golden']['heatmap_bit_identical'] and d['golden']['identity_equal'] and not d['network_reachable']" "$OUT/selfcheck.json"

# ------------------------------------------------------------------- 3. compose deployment
log "== 3. compose up"
ENVF=$OUT/compose.env
cat > "$ENVF" <<EOF
BEVSERVE_IMAGE=$IMAGE
BEVSERVE_CHECKPOINT_DIR=$CKPT_DIR
BEVSERVE_CHECKPOINT_FILE=$CKPT_FILE
BEVSERVE_CHECKPOINT_SHA256=$CKPT_SHA256
BEVSERVE_HOST_PORT=$PORT
BEVSERVE_GPU=$GPU
BEVSERVE_GIT_COMMIT=$(git -C "$REPO" describe --always --dirty --abbrev=7)
EOF
BASE_DC=(compose -p "$PROJECT" -f "$REPO/docker/compose.yaml" --env-file "$ENVF")
DC=("${BASE_DC[@]}")
COMPOSE_USED=1
d "${DC[@]}" up -d --no-build >> "$LOG" 2>&1 || die "compose up"
CID=$(d "${DC[@]}" ps -q bevserve)
[ -n "$CID" ] || die "compose container id"
echo "first /readyz during startup: $(readyz)" >> "$LOG"
secs=$(wait_ready 240) && pass "startup-ready (${secs} s)" \
  || { echo "last /readyz: $(readyz)" >> "$LOG"; die "startup-ready (see compose-logs.txt)"; }
POLICY=$(d inspect -f '{{.HostConfig.RestartPolicy.Name}}:{{.HostConfig.RestartPolicy.MaximumRetryCount}}' "$CID")
[ "$POLICY" = "on-failure:5" ] && pass "restart-policy ($POLICY)" || fail "restart-policy ($POLICY)"
PUBLISHED=$(d port "$CID" 8000/tcp); echo "published: $PUBLISHED" >> "$LOG"
if [ -n "$PUBLISHED" ] && ! grep -v -q -E "^127\.0\.0\.1:$PORT$" <<< "$PUBLISHED"; then pass "port-published-on-loopback-only ($PUBLISHED)"
else fail "port-published-on-loopback-only ($PUBLISHED)"; fi
LISTEN=$(ss -ltnH "( sport = :$PORT )" | awk '{print $4}'); echo "host listeners: $LISTEN" >> "$LOG"
if [ -n "$LISTEN" ] && ! grep -v -q -E '^(127\.0\.0\.1|\[::1\]):' <<< "$LISTEN"; then pass "host-listener-loopback-only"; else fail "host-listener-loopback-only"; fi
curl -s "http://127.0.0.1:$PORT/v1/model" > "$OUT/model.json"
check "model-identity" json_ok "d['checkpoint_sha256'] == sys.argv[2] and d['frame'] == 'ego@CAM_FRONT_LEFT'" "$OUT/model.json" "$CKPT_SHA256"
d stats --no-stream --format '{{.Name}} cpu={{.CPUPerc}} mem={{.MemUsage}}' "$CID" >> "$LOG" 2>&1
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >> "$LOG"

# ------------------------------------------------------------------- 4. real-frame parity
log "== 4. real-frame parity vs M3 golden (host client; golden never enters a container)"
( cd "$BEVDET_ROOT" && PYTHONDONTWRITEBYTECODE=1 "$HOST_PY" "$REPO/tools/golden_http.py" compare \
    --url "http://127.0.0.1:$PORT" --golden "$GOLDEN_DIR/http_golden.json" ) > "$OUT/golden_compare.json" 2> "$OUT/golden_compare.stderr"
check "http-golden-parity (10 real frames, byte-identical boxes)" json_ok "d['pass']" "$OUT/golden_compare.json"

# M6: the same golden comparison with the experimental parallel preprocessing enabled. The env
# file gets the setting only for this step and is restored before the fault test.
PARALLEL_THREADS=${PARALLEL_THREADS:-12}
log "== 4b. real-frame parity with BEVSERVE_PREPROCESS_THREADS=$PARALLEL_THREADS"
full_id() { local c; c=$(d "${DC[@]}" ps -q bevserve 2>> "$LOG") && [ -n "$c" ] && d inspect -f '{{.Id}}' "$c" 2>> "$LOG"; }
threads_env() {  # the running container's own BEVSERVE_PREPROCESS_THREADS (Config.Env)
  d inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$1" 2>> "$LOG" | sed -n 's/^BEVSERVE_PREPROCESS_THREADS=//p'
}
GOLDEN_FRAMES=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))['frames']))" "$GOLDEN_DIR/http_golden.json")
SERIAL_CID=$(full_id) || SERIAL_CID=
SERIAL_THREADS=$( [ -n "$SERIAL_CID" ] && threads_env "$SERIAL_CID" )
echo "serial container $SERIAL_CID: BEVSERVE_PREPROCESS_THREADS=${SERIAL_THREADS:-unset}" >> "$LOG"
if [ -z "$SERIAL_CID" ] || [ "$SERIAL_THREADS" != 0 ]; then
  fail "parallel: serial container not verified (id ${SERIAL_CID:-none}, threads ${SERIAL_THREADS:-unset}); parallel check not run"
else
  pass "parallel: serial container runs with BEVSERVE_PREPROCESS_THREADS=0"
  cp "$ENVF" "$ENVF.serial"
  echo "BEVSERVE_PREPROCESS_THREADS=$PARALLEL_THREADS" >> "$ENVF"
  PAR_CID= PAR_THREADS=
  if d "${DC[@]}" up -d --no-build >> "$LOG" 2>&1 && PAR_CID=$(full_id); then PAR_THREADS=$(threads_env "$PAR_CID"); fi
  echo "parallel container ${PAR_CID:-none}: BEVSERVE_PREPROCESS_THREADS=${PAR_THREADS:-unset}" >> "$LOG"
  if [ -z "$PAR_CID" ] || [ "$PAR_CID" = "$SERIAL_CID" ] || [ "$PAR_THREADS" != "$PARALLEL_THREADS" ]; then
    fail "parallel: container not recreated with BEVSERVE_PREPROCESS_THREADS=$PARALLEL_THREADS (id ${PAR_CID:-none}, threads ${PAR_THREADS:-unset})"
  elif ! wait_ready 240 > /dev/null; then
    fail "parallel: service did not become ready"
  else
    pass "parallel: recreated container (new id) runs with BEVSERVE_PREPROCESS_THREADS=$PARALLEL_THREADS"
    ( cd "$BEVDET_ROOT" && PYTHONDONTWRITEBYTECODE=1 "$HOST_PY" "$REPO/tools/golden_http.py" compare \
        --url "http://127.0.0.1:$PORT" --golden "$GOLDEN_DIR/http_golden.json" ) \
      > "$OUT/golden_compare_parallel.json" 2> "$OUT/golden_compare_parallel.stderr"
    check "http-golden-parity with preprocess threads $PARALLEL_THREADS ($GOLDEN_FRAMES real frames, byte-identical boxes)" \
      json_ok "d['pass'] and d['model_identity_ok'] and len(d['frames']) == int(sys.argv[2]) == 10 and all(f['status'] == 200 and f['boxes_identical'] and f['frame_units_ok'] and f['identity_ok'] for f in d['frames'])" \
      "$OUT/golden_compare_parallel.json" "$GOLDEN_FRAMES"
  fi
  mv "$ENVF.serial" "$ENVF"   # always restored; later steps recreate from the serial env file
  grep -q '^BEVSERVE_PREPROCESS_THREADS=' "$ENVF" && fail "parallel: serial env file not restored"
fi

# ------------------------------------------------------------------- 5. fatal exit and restart
log "== 5. fatal runtime error -> exit 70 -> supervised restart -> ready"
d "${DC[@]}" stop >> "$LOG" 2>&1
cat > "$OUT/compose.fault.yaml" <<'EOF'
services:
  bevserve:
    entrypoint: ["python", "-m", "bevserve.faultinject", "--fatal-on-call", "2"]
    environment:
      BEVSERVE_FAULT_INJECTION_ACK: "1"
EOF
DC=("${BASE_DC[@]}" -f "$OUT/compose.fault.yaml"); FDC=("${DC[@]}")
d "${FDC[@]}" up -d --no-build >> "$LOG" 2>&1 || fail "fault: compose up"
FID=$(d "${FDC[@]}" ps -q bevserve) && [ -n "$FID" ] && FID=$(d inspect -f '{{.Id}}' "$FID")   # full id (events)
if [ -n "$FID" ] && [ "$(threads_env "$FID")" = 0 ]; then pass "fault: container runs with BEVSERVE_PREPROCESS_THREADS=0 (env restored)"
else fail "fault: container not running with BEVSERVE_PREPROCESS_THREADS=0"; fi
if [ -n "$FID" ] && wait_ready 240 > /dev/null; then
  T_START=$(date +%s.%N); RC0=$(d inspect -f '{{.RestartCount}}' "$FID")
  PYTHONPATH="$REPO" PYTHONDONTWRITEBYTECODE=1 "$HOST_PY" - "$PORT" > "$OUT/fault_requests.txt" 2>> "$LOG" <<'PY'
import sys, time, httpx
from tests.synthetic import multipart_files
port, seen = sys.argv[1], []
files, data = multipart_files()
for _ in range(6):
    try:
        r = httpx.post(f"http://127.0.0.1:{port}/v1/detect", files=files, data=data, timeout=30)
        seen.append(f"{r.status_code}:{r.json().get('type', 'ok')}")
    except httpx.HTTPError as exc:
        seen.append(type(exc).__name__)
    time.sleep(0.2)
print(" ".join(seen))
PY
  cat "$OUT/fault_requests.txt" >> "$LOG"
  check "fault: call 1 ok, call 2 -> 500 inference_fatal" grep -q -E '^200:ok 500:/errors/inference_fatal' "$OUT/fault_requests.txt"
  t0=$SECONDS
  while (( SECONDS - t0 < 120 )) && [ "$(d inspect -f '{{.RestartCount}}' "$FID")" = "$RC0" ]; do sleep 1; done
  RC1=$(d inspect -f '{{.RestartCount}}' "$FID")
  # --until must lie in the future: the daemon drops events later than --until, and a
  # whole-second "now" cut off a die event in the current second (M5 stage A: empty result).
  d events --since "$T_START" --until "$(( $(date +%s) + 2 ))" --filter type=container \
    --filter "label=com.docker.compose.project=$PROJECT" --format '{{json .}}' > "$OUT/fault_events.jsonl" 2>> "$LOG"
  if python3 "$REPO/tools/fault_events.py" --events "$OUT/fault_events.jsonl" --container "$FID" --exit-code 70 \
       --restarts-before "$RC0" --restarts-after "$RC1" > "$OUT/fault_events.json" 2>> "$LOG"; then
    pass "fault: process exited with code 70 (docker die event; restart $RC0 -> $RC1)"
  else
    fail "fault: process exited with code 70 (see fault_events.json)"
  fi
  cat "$OUT/fault_events.json" >> "$LOG"
  [ "$RC1" -gt "$RC0" ] && pass "fault: docker restarted the container ($RC0 -> $RC1)" || fail "fault: docker restarted the container"
  d logs "$FID" 2>&1 | grep -q "exiting with code 70" && pass "fault: fatal exit logged" || fail "fault: fatal exit logged"
  secs=$(wait_ready 240) && pass "fault: ready again after restart (${secs} s)" || fail "fault: ready again after restart"
else
  fail "fault: injected service did not become ready"
fi
d "${FDC[@]}" down --remove-orphans >> "$LOG" 2>&1
DC=("${BASE_DC[@]}")

# ------------------------------------------------------------------- 6. initialization failures
log "== 6. initialization failures: service stays up, not ready, no restarts"
init_case() {  # init_case <name> <reason> <docker run args (image last)...>
  local name=$1 reason=$2 cid body t0; shift 2
  run_labeled -p "127.0.0.1:$PORT:8000" --restart on-failure:5 "$@" || { fail "$name: container start"; return; }
  cid=$LAST_CID
  t0=$SECONDS; body=""
  while (( SECONDS - t0 < 180 )); do
    body=$(readyz) && grep -q '"init_failed"' <<< "$body" && break; sleep 1
  done
  sleep 10   # long enough for a crash loop to show up in RestartCount
  local running restarts health detect
  running=$(d inspect -f '{{.State.Running}}' "$cid"); restarts=$(d inspect -f '{{.RestartCount}}' "$cid")
  health=$(http_code "http://127.0.0.1:$PORT/healthz"); detect=$(http_code -X POST -F x=1 "http://127.0.0.1:$PORT/v1/detect")
  echo "$name: readyz=$body running=$running restarts=$restarts healthz=$health detect=$detect" >> "$LOG"
  if grep -q "\"reason\":\"$reason\"" <<< "$body" && [ "$running" = true ] && [ "$restarts" = 0 ] \
     && [ "$health" = 200 ] && [ "$detect" = 503 ]; then pass "$name ($reason; up, not ready, 0 restarts)"
  else fail "$name (expected $reason)"; fi
  remove_container "$cid"
}
COMMON=(-v "$CKPT_DIR:/models:ro" -e "BEVSERVE_CHECKPOINT=/models/$CKPT_FILE")
init_case "wrong-checkpoint-hash" checkpoint_hash_mismatch --gpus "device=$GPU" "${COMMON[@]}" \
  -e "BEVSERVE_CHECKPOINT_SHA256=$(printf '0%.0s' {1..64})" "$IMAGE"
init_case "missing-gpu" cuda_unavailable "${COMMON[@]}" -e "BEVSERVE_CHECKPOINT_SHA256=$CKPT_SHA256" "$IMAGE"
BLOCKED_SERVER="import sys; from bevserve.selfcheck import _Blocker; sys.meta_path.insert(0, _Blocker('mmdet3d.ops.bev_pool_v2.bev_pool_v2_ext')); from bevserve.cli import main; sys.exit(main())"
init_case "missing-custom-op (running service)" model_stack_missing --gpus "device=$GPU" "${COMMON[@]}" \
  -e "BEVSERVE_CHECKPOINT_SHA256=$CKPT_SHA256" --entrypoint python "$IMAGE" -c "$BLOCKED_SERVER"

# ------------------------------------------------------------------- 7. clean shutdown
log "== 7. clean startup and shutdown"
d "${DC[@]}" up -d --no-build >> "$LOG" 2>&1 || fail "shutdown: compose up"
CID=$(d "${DC[@]}" ps -q bevserve)
if [ -n "$CID" ] && wait_ready 240 > /dev/null; then
  t0=$SECONDS; d "${DC[@]}" stop >> "$LOG" 2>&1
  code=$(d inspect -f '{{.State.ExitCode}}' "$CID")
  [ "$code" = 0 ] && pass "clean-shutdown (exit 0 in $((SECONDS - t0)) s)" || fail "clean-shutdown (exit $code)"
else
  fail "shutdown: service did not become ready"
fi
d "${DC[@]}" down --remove-orphans >> "$LOG" 2>&1 && COMPOSE_USED=0
log "done"
