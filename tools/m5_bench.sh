#!/usr/bin/env bash
# M5/M6 performance characterization of a verified BEVServe GPU image. Operator-run.
#
#   MODE=pilot|full IMAGE=bevserve:<tag> EXPECT_IMAGE_ID=sha256:<id> \
#   STAGE_A_DIR=outputs/m4-<run> FAULT_RETEST_DIR=outputs/m5-fault-<run> \
#   BEVDET_ROOT=... CKPT_DIR=... CKPT_SHA256=... GOLDEN_DIR=... HOST_PY=... bash tools/m5_bench.sh
#
# MODE=pilot validates the harness on the real GPU with bounded requests (all stages, concurrency
# 1, 2, 4 and 8; statistical criteria reported but not counted). MODE=full runs the defaults below.
# STAGES (default: all) selects among:
#   cold       COLD_RUNS x: start -> /readyz 200 time, first request, stop
#   levels     server timing on, LEVELS_MAX_IN_FLIGHT: C = 1, 2, 4, 8, LEVEL_REQUESTS each after
#              LEVEL_WARMUP warm-up requests; resource sampling
#   overhead   OVERHEAD_TRIALS fresh services in ABBA order (A = timing off, B = on), C = 1,
#              OVERHEAD_REQUESTS each after OVERHEAD_WARMUP; paired t interval (tools/bench/overhead.py)
#   admission  production MAX_IN_FLIGHT 5: C in ADMISSION_LEVELS, with and without honoring
#              Retry-After, ADMISSION_REQUESTS each; then readiness and 5 recovery requests; sampling
#   soak       C = 2, SOAK_REQUESTS (100 = pilot soak; > 100 needs SOAK_FULL=1, max 1000); sampling,
#              then /readyz and the 10-frame golden parity comparison
#   ab         (M6) AB_TRIALS fresh services in ABBA order, A = BEVSERVE_PREPROCESS_THREADS=0,
#              B = AB_THREADS; identical settings otherwise (server timing AB_SERVER_TIMING,
#              max_in_flight AB_MAX_IN_FLIGHT). Per trial: C = 1 (AB_C1_REQUESTS) and C = 8
#              (AB_C8_REQUESTS), each after AB_WARMUP warm-up, then the 10-frame golden parity;
#              one 0.5 s resource sampler per trial; analysis tools/bench/ab.py. MODE=full needs
#              PILOT_DIR = a passing MODE=pilot ab run on the same image and AB_THREADS.
#              Without FAULT_RETEST_DIR (M6 images) the verification run must be all PASS and
#              include the parallel golden and container-recreation checks for AB_THREADS.
#
#                   pilot  full (default)  allowed
#   COLD_RUNS       1      3               1-5
#   LEVEL_REQUESTS  20     300             1-1000 (pilot <= 50); LEVEL_WARMUP 5 / 20 (0-100)
#   OVERHEAD_TRIALS 4      12              multiple of 4, 4-16; OVERHEAD_REQUESTS 20 / 100 (1-500)
#   ADMISSION_LEVELS "8"   "8 16"          8 and/or 16; ADMISSION_REQUESTS 40 / 200 (1-500)
#   SOAK_REQUESTS   20     100             pilot <= 50; full <= 100 unless SOAK_FULL=1 (<= 1000)
#   AB_TRIALS       4      12              multiple of 4, 4-16; AB_THREADS 12 (1-32)
#   AB_C1_REQUESTS  20     100             1-500 (pilot <= 50); AB_C8_REQUESTS 40 / 200 (1-500)
#   AB_WARMUP       5      20              0-100; AB_MAX_IN_FLIGHT 16 (8-32)
#
# Safety: requires the Stage A image id (EXPECT_IMAGE_ID, checked against the Stage A record and the
# passing fault retest) and an image whose bevserve package equals this work tree. One Compose
# project per run (bevserve-m5-<run id>); every compose call uses the same -p/-f/--env-file; only
# that project is ever removed, and removal is verified (no containers or network left); a failed
# cleanup is reported and makes the exit status non-zero (2). 127.0.0.1 only; private outputs
# (umask 077) in outputs/m5-<run id>/; no build; no host changes; refuses a busy port or GPU.
set -uo pipefail
umask 077

REPO=$(cd "$(dirname "$0")/.." && pwd)
RUN_ID=$(date +%Y%m%d-%H%M%S)
PROJECT=bevserve-m5-$RUN_ID
PORT=${PORT:-8000}
GPU=${GPU:-0}
CKPT_FILE=${CKPT_FILE:-bevdet-r50-cbgs.pth}
MODE=${MODE:-}
STAGES=${STAGES:-cold levels overhead admission soak}
INPUTS=${INPUTS:-nuscenes:even:20}
NVIDIA_SMI=${NVIDIA_SMI:-nvidia-smi}
# Golden comparison tool. Overriding it is for the fake-Docker tests only: it needs
# BENCH_TEST_MODE=1, is recorded in meta.json, and a test-mode pilot never authorizes a full run.
GOLDEN_TOOL=${GOLDEN_TOOL:-$REPO/tools/golden_http.py}
TEST_MODE=${BENCH_TEST_MODE:-0}
export NVIDIA_SMI
read -r -a DOCKER <<< "${DOCKER_CMD:-sudo docker}"
d() { "${DOCKER[@]}" "$@"; }
OUT=$REPO/outputs/m5-$RUN_ID
mkdir -p "$OUT" && chmod 700 "$OUT"
LOG=$OUT/bench.log RUNS=$OUT/runs.jsonl ENVF=$OUT/compose.env
: > "$LOG"; : > "$RUNS"
DC=(compose -p "$PROJECT" -f "$REPO/docker/compose.yaml" --env-file "$ENVF")   # every compose call
exec 3>&1 4>&2
FAILED=0
UP=0
CLEANUP_FAILED=0
SERVICE_SEQ=0
SAMPLER= SAMPLER_LABEL= SAMPLER_T0=
SUDO_KEEPALIVE=

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
die() { log "ABORT: $*"; exit 1; }
record() { echo "$1" >> "$RUNS"; }   # one JSON object per line (built from plain tokens only)

cleanup() {
  local code=$?
  trap - EXIT INT TERM
  exec 1>&3 2>&4
  stop_sampler
  service_down || true
  [ -n "$SUDO_KEEPALIVE" ] && kill "$SUDO_KEEPALIVE" 2>/dev/null
  if [ "$CLEANUP_FAILED" = 1 ]; then
    log "RESULT: CLEANUP FAILED - Compose project $PROJECT may still exist. Inspect it, then remove only it:"
    log "  ${DOCKER[*]} ${DC[*]} down --remove-orphans"
    log "outputs: $OUT"
    exit 2
  fi
  if [ "$code" -ne 0 ]; then log "RESULT: run aborted (exit $code); outputs: $OUT"; exit 1; fi
  if [ "$FAILED" -ne 0 ]; then log "RESULT: acceptance criteria not met (see criteria.json); outputs: $OUT"; exit 1; fi
  log "RESULT: all counted criteria passed ($MODE); outputs: $OUT (report.md, criteria.json)"
  exit 0
}
trap cleanup EXIT
trap 'exec 1>&3 2>&4; log "interrupted"; exit 130' INT TERM

# ------------------------------------------------------------------- parameters and bounds
case "$MODE" in
  pilot) P_COLD=1 P_LR=20 P_LW=5 P_OT=4 P_OR=20 P_OW=5 P_AL="8" P_AR=40 P_SR=20 P_SW=5 CAP=50
         P_ABT=4 P_AB1=20 P_AB8=40 P_ABW=5 ;;
  full)  P_COLD=3 P_LR=300 P_LW=20 P_OT=12 P_OR=100 P_OW=20 P_AL="8 16" P_AR=200 P_SR=100 P_SW=20 CAP=1000
         P_ABT=12 P_AB1=100 P_AB8=200 P_ABW=20 ;;
  *) die "MODE must be pilot or full" ;;
esac
COLD_RUNS=${COLD_RUNS:-$P_COLD}
LEVEL_REQUESTS=${LEVEL_REQUESTS:-$P_LR}
LEVEL_WARMUP=${LEVEL_WARMUP:-$P_LW}
LEVELS_MAX_IN_FLIGHT=${LEVELS_MAX_IN_FLIGHT:-16}
OVERHEAD_TRIALS=${OVERHEAD_TRIALS:-$P_OT}
OVERHEAD_REQUESTS=${OVERHEAD_REQUESTS:-$P_OR}
OVERHEAD_WARMUP=${OVERHEAD_WARMUP:-$P_OW}
ADMISSION_LEVELS=${ADMISSION_LEVELS:-$P_AL}
ADMISSION_REQUESTS=${ADMISSION_REQUESTS:-$P_AR}
SOAK_REQUESTS=${SOAK_REQUESTS:-$P_SR}
SOAK_WARMUP=${SOAK_WARMUP:-$P_SW}
AB_TRIALS=${AB_TRIALS:-$P_ABT}
AB_THREADS=${AB_THREADS:-12}
AB_C1_REQUESTS=${AB_C1_REQUESTS:-$P_AB1}
AB_C8_REQUESTS=${AB_C8_REQUESTS:-$P_AB8}
AB_WARMUP=${AB_WARMUP:-$P_ABW}
AB_MAX_IN_FLIGHT=${AB_MAX_IN_FLIGHT:-16}
AB_SERVER_TIMING=${AB_SERVER_TIMING:-true}
bounded() {  # bounded NAME MIN MAX: integer within [MIN, MAX] or abort
  local v=${!1}
  [[ "$v" =~ ^[0-9]+$ ]] && (( 10#$v >= $2 && 10#$v <= $3 )) || die "$1=$v must be an integer in [$2, $3]"
}
SMALL_CAP=$(( CAP < 500 ? CAP : 500 ))
bounded COLD_RUNS 1 5
bounded LEVEL_REQUESTS 1 "$CAP"
bounded LEVEL_WARMUP 0 100
bounded LEVELS_MAX_IN_FLIGHT 8 32
bounded OVERHEAD_TRIALS 4 16
(( OVERHEAD_TRIALS % 4 == 0 )) || die "OVERHEAD_TRIALS must be a multiple of 4 (whole ABBA blocks)"
bounded OVERHEAD_REQUESTS 1 "$SMALL_CAP"
bounded OVERHEAD_WARMUP 0 100
bounded ADMISSION_REQUESTS 1 "$SMALL_CAP"
bounded SOAK_WARMUP 0 100
bounded AB_TRIALS 4 16
(( AB_TRIALS % 4 == 0 )) || die "AB_TRIALS must be a multiple of 4 (whole ABBA blocks)"
bounded AB_THREADS 1 32
bounded AB_C1_REQUESTS 1 "$SMALL_CAP"
bounded AB_C8_REQUESTS 1 "$SMALL_CAP"
bounded AB_WARMUP 0 100
bounded AB_MAX_IN_FLIGHT 8 32
case "$AB_SERVER_TIMING" in true|false) ;; *) die "AB_SERVER_TIMING must be true or false" ;; esac
[ -n "$ADMISSION_LEVELS" ] || die "ADMISSION_LEVELS is empty"
for c in $ADMISSION_LEVELS; do case "$c" in 8|16) ;; *) die "ADMISSION_LEVELS may contain only 8 and 16" ;; esac; done
bounded SOAK_REQUESTS 1 "$CAP"
if (( SOAK_REQUESTS > 100 )) && [ "${SOAK_FULL:-0}" != 1 ]; then
  die "SOAK_REQUESTS=$SOAK_REQUESTS needs SOAK_FULL=1 (separate authorization for the full soak)"
fi
for stage in $STAGES; do
  case "$stage" in cold|levels|overhead|admission|soak|ab) ;; *) die "unknown stage $stage" ;; esac
done
if [ "$GOLDEN_TOOL" != "$REPO/tools/golden_http.py" ]; then
  [ "$TEST_MODE" = 1 ] || die "GOLDEN_TOOL may only be overridden with BENCH_TEST_MODE=1 (tests)"
  log "WARNING: TEST MODE - golden comparison by $GOLDEN_TOOL; results are not benchmark evidence"
fi

# ------------------------------------------------------------------- inputs and prerequisites
for var in IMAGE EXPECT_IMAGE_ID STAGE_A_DIR BEVDET_ROOT CKPT_DIR CKPT_SHA256 GOLDEN_DIR HOST_PY; do
  [ -n "${!var:-}" ] || die "$var is not set"
done
FAULT_RETEST_DIR=${FAULT_RETEST_DIR:-}
DIRS=(STAGE_A_DIR BEVDET_ROOT CKPT_DIR GOLDEN_DIR)
[ -n "$FAULT_RETEST_DIR" ] && DIRS+=(FAULT_RETEST_DIR)
for var in "${DIRS[@]}"; do
  abs=$(cd "${!var}" 2>/dev/null && pwd) || die "$var is not a directory"
  printf -v "$var" '%s' "$abs"
done
AB=0; [[ " $STAGES " == *" ab "* ]] && AB=1
if [ "$AB" = 1 ] && [ "$MODE" = full ]; then
  [ -n "${PILOT_DIR:-}" ] || die "MODE=full with stage ab needs PILOT_DIR (a passing MODE=pilot ab run)"
  PILOT_DIR=$(cd "$PILOT_DIR" 2>/dev/null && pwd) || die "PILOT_DIR is not a directory"
fi
case "$HOST_PY" in /*) ;; *) HOST_PY=$(realpath -s "$HOST_PY") ;; esac
CKPT_ACTUAL=$(sha256sum "$CKPT_DIR/$CKPT_FILE" | cut -d' ' -f1)
[ "$CKPT_ACTUAL" = "$CKPT_SHA256" ] || die "checkpoint sha256 mismatch"
[ -r "$GOLDEN_DIR/http_golden.json" ] || die "golden missing"
"$HOST_PY" -c "import httpx, numpy" > /dev/null 2>&1 || die "HOST_PY lacks httpx/numpy"
if [[ "$INPUTS" == nuscenes:* ]]; then "$HOST_PY" -c "import mmcv" > /dev/null 2>&1 || die "HOST_PY lacks mmcv"; fi
if [[ "${DOCKER[0]}" == sudo ]]; then
  sudo -v || die "sudo authentication"
  ( while sleep 60; do sudo -n -v 2>/dev/null || exit; done ) & SUDO_KEEPALIVE=$!
fi
IMAGE_ID=$(d image inspect -f '{{.Id}}' "$IMAGE" 2>> "$LOG") || die "image $IMAGE not found"
GATE=(--expect "$EXPECT_IMAGE_ID" --actual "$IMAGE_ID" --stage-a "$STAGE_A_DIR")
if [ -n "$FAULT_RETEST_DIR" ]; then GATE+=(--retest "$FAULT_RETEST_DIR"); fi
if [ "$AB" = 1 ]; then
  GATE+=(--require "http-golden-parity with preprocess threads $AB_THREADS ("
         --require "recreated container (new id) runs with BEVSERVE_PREPROCESS_THREADS=$AB_THREADS")
fi
python3 "$REPO/tools/bench/verified_image.py" "${GATE[@]}" > "$OUT/verified-image.json" \
  || die "image is not the verified image (verified-image.json)"
if [ "$AB" = 1 ] && [ "$MODE" = full ]; then
  python3 "$REPO/tools/bench/ab.py" pilot-gate --pilot "$PILOT_DIR" --image-id "$IMAGE_ID" \
    --threads "$AB_THREADS" > "$OUT/pilot-gate.json" || die "pilot did not pass (pilot-gate.json)"
fi
d image inspect -f '{{json .}}' "$IMAGE" 2>> "$LOG" | python3 -c "
import json, sys
i = json.load(sys.stdin)
json.dump({k: i.get(k) for k in ('Id', 'Created', 'Size', 'RepoTags')} | {'Labels': (i.get('Config') or {}).get('Labels')}, sys.stdout, indent=2)
" > "$OUT/image-inspect.json" || die "image inspect"
python3 "$REPO/tools/provenance.py" source --repo "$REPO" > "$OUT/provenance-source.json" || die "source provenance"
d run --rm -i --network none --entrypoint python "$IMAGE" - package < "$REPO/tools/provenance.py" \
  > "$OUT/provenance-image.json" 2>> "$LOG" || die "image package provenance"
python3 "$REPO/tools/provenance.py" compare "$OUT/provenance-source.json" "$OUT/provenance-image.json" \
  > "$OUT/provenance-compare.json" || die "bevserve package in $IMAGE differs from the work tree (provenance-compare.json)"
GIT_DESCRIBE=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['git_describe'])" "$OUT/provenance-source.json")
ss -ltnH "( sport = :$PORT )" | grep -q . && die "port $PORT already in use"
[ -z "$(d ps -aq --filter "label=com.docker.compose.project=$PROJECT")" ] || die "project $PROJECT exists"
BUSY=$("$NVIDIA_SMI" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' '$2 > 1024')
[ -z "$BUSY" ] || die "another GPU workload is running ($BUSY)"

"$HOST_PY" - "$OUT" <<PY || die "meta.json"
import hashlib, json, os, platform, subprocess, sys
out = sys.argv[1]
def sh(*cmd):
    try: return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception: return ""
def sha(path): return hashlib.sha256(open(path, "rb").read()).hexdigest()
def load(name): return json.load(open(os.path.join(out, name)))
cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "")
mem_gib = round(int(next(l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemTotal"))) / 2**20)
harness = ["tools/m5_bench.sh", "docker/compose.yaml", "tools/provenance.py", "tools/golden_http.py",
           *sorted(f"tools/bench/{f}" for f in os.listdir("$REPO/tools/bench") if f.endswith(".py"))]
meta = {
  "run_id": "$RUN_ID", "mode": "$MODE",
  "image": {"tag": "$IMAGE", **load("image-inspect.json")},
  "verified_image": load("verified-image.json"),
  "source": {k: v for k, v in load("provenance-source.json").items() if k != "package_files"},
  "image_package_matches_source": load("provenance-compare.json")["match"],
  "harness_sha256": {f: sha(os.path.join("$REPO", f)) for f in harness},
  "checkpoint": {"file": "$CKPT_FILE", "sha256": "$CKPT_ACTUAL",
                 "size_bytes": os.path.getsize("$CKPT_DIR/$CKPT_FILE")},
  "golden_sha256": sha("$GOLDEN_DIR/http_golden.json"),
  "golden_tool": {"path": os.path.relpath("$GOLDEN_TOOL", "$REPO"), "sha256": sha("$GOLDEN_TOOL")},
  "test_mode": "$TEST_MODE" == "1",
  "gpu": sh("$NVIDIA_SMI", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"),
  "kernel": platform.release(), "host_cpu": cpu, "host_cpus": os.cpu_count(), "host_mem_gib": mem_gib,
  "inputs": "$INPUTS", "production_max_in_flight": 5, "deadline_s": 10,
  "client": "tools/bench/load.py (closed loop, same host)",
  "params": {
    "mode": "$MODE", "stages": "$STAGES".split(), "cold_runs": $COLD_RUNS,
    "level_requests": $LEVEL_REQUESTS, "level_warmup": $LEVEL_WARMUP,
    "levels_max_in_flight": $LEVELS_MAX_IN_FLIGHT, "overhead_trials": $OVERHEAD_TRIALS,
    "overhead_requests": $OVERHEAD_REQUESTS, "overhead_warmup": $OVERHEAD_WARMUP,
    "admission_levels": [int(c) for c in "$ADMISSION_LEVELS".split()],
    "admission_requests": $ADMISSION_REQUESTS, "soak_requests": $SOAK_REQUESTS,
    "soak_warmup": $SOAK_WARMUP, "soak_full": "${SOAK_FULL:-0}" == "1",
    "ab_trials": $AB_TRIALS, "ab_threads": $AB_THREADS, "ab_c1_requests": $AB_C1_REQUESTS,
    "ab_c8_requests": $AB_C8_REQUESTS, "ab_warmup": $AB_WARMUP,
    "ab_max_in_flight": $AB_MAX_IN_FLIGHT, "ab_server_timing": "$AB_SERVER_TIMING" == "true",
    "pilot_dir": "${PILOT_DIR:+$(basename "${PILOT_DIR:-}")}",
  },
}
json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)
PY
log "run $RUN_ID ($MODE): image $IMAGE ($IMAGE_ID), source $GIT_DESCRIBE, stages: $STAGES"

# ------------------------------------------------------------------- helpers
LAST_CID=
service_up() {  # service_up <stage> <server_timing true|false> <max_in_flight> [preprocess threads]
  local threads=${4:-0}
  cat > "$ENVF" <<EOF
BEVSERVE_IMAGE=$IMAGE
BEVSERVE_CHECKPOINT_DIR=$CKPT_DIR
BEVSERVE_CHECKPOINT_FILE=$CKPT_FILE
BEVSERVE_CHECKPOINT_SHA256=$CKPT_SHA256
BEVSERVE_HOST_PORT=$PORT
BEVSERVE_GPU=$GPU
BEVSERVE_SERVER_TIMING=$2
BEVSERVE_MAX_IN_FLIGHT=$3
BEVSERVE_PREPROCESS_THREADS=$threads
BEVSERVE_GIT_COMMIT=$GIT_DESCRIBE
EOF
  UP=1   # from here on cleanup must remove the project, even if `up` fails half-way
  SERVICE_SEQ=$((SERVICE_SEQ + 1))
  T_UP=$(date +%s.%N)
  d "${DC[@]}" up -d --no-build >> "$LOG" 2>&1 || die "compose up"
  local i
  for i in $(seq 1 1200); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:$PORT/readyz")" = 200 ] && break
    sleep 0.2
  done
  [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/readyz")" = 200 ] || die "service not ready"
  READY_S=$(python3 -c "import time; print(round(time.time() - $T_UP, 2))")
  CID=$(d "${DC[@]}" ps -q bevserve) && [ -n "$CID" ] && CID=$(d inspect -f '{{.Id}}' "$CID") || die "service container id"
  [ "$CID" != "$LAST_CID" ] || die "service container was not recreated (same id as the previous service)"
  LAST_CID=$CID
  ENV_THREADS=$(d inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$CID" | sed -n 's/^BEVSERVE_PREPROCESS_THREADS=//p')
  [ "$ENV_THREADS" = "$threads" ] || die "container runs with BEVSERVE_PREPROCESS_THREADS=${ENV_THREADS:-unset}, expected $threads"
  SERVER_PID=$(d inspect -f '{{.State.Pid}}' "$CID") && [[ "$SERVER_PID" =~ ^[1-9][0-9]*$ ]] || die "server pid"
  record "{\"kind\": \"service\", \"seq\": $SERVICE_SEQ, \"stage\": \"$1\", \"server_timing\": $2, \"max_in_flight\": $3, \"preprocess_threads\": $threads, \"preprocess_threads_env\": $ENV_THREADS, \"ready_s\": $READY_S, \"container\": \"$CID\", \"host_pid\": $SERVER_PID}"
  log "service $SERVICE_SEQ up ($1: timing=$2, max_in_flight=$3, preprocess_threads=$threads) ready in ${READY_S} s, host pid $SERVER_PID"
}
service_down() {  # removes this run's project; UP stays 1 unless the removal is verified
  [ "$UP" = 1 ] || return 0
  d "${DC[@]}" logs --no-color --timestamps >> "$OUT/compose-logs.txt" 2>&1 || log "WARNING: compose logs not saved"
  local rc=0 left nets
  d "${DC[@]}" down --remove-orphans >> "$LOG" 2>&1 || rc=$?
  left=$(d ps -aq --filter "label=com.docker.compose.project=$PROJECT" 2>> "$LOG") || left="query-failed"
  nets=$(d network ls -q --filter "label=com.docker.compose.project=$PROJECT" 2>> "$LOG") || nets="query-failed"
  left=$(tr '\n' ' ' <<< "$left" | xargs) nets=$(tr '\n' ' ' <<< "$nets" | xargs)
  if [ "$rc" = 0 ] && [ -z "$left" ] && [ -z "$nets" ]; then
    UP=0 CLEANUP_FAILED=0
    record "{\"kind\": \"cleanup\", \"seq\": $SERVICE_SEQ, \"ok\": true}"
    return 0
  fi
  CLEANUP_FAILED=1
  log "CLEANUP FAILED for $PROJECT: down exit $rc; containers left: ${left:-none}; networks left: ${nets:-none}"
  record "{\"kind\": \"cleanup\", \"seq\": $SERVICE_SEQ, \"ok\": false, \"detail\": \"down exit $rc; containers: ${left:-none}; networks: ${nets:-none}\"}"
  return 1
}
start_sampler() {  # start_sampler <label> [interval s]
  SAMPLER_LABEL=$1 SAMPLER_T0=$(date +%s.%N)
  "$HOST_PY" "$REPO/tools/bench/sampler.py" --pid "$SERVER_PID" --out "$OUT/$1.samples.csv" \
    --meta "$OUT/$1.sampler.json" --interval "${2:-1}" 2>> "$LOG" &
  SAMPLER=$!
}
stop_sampler() {  # a sampler that died early or exits non-zero yields an invalid record
  [ -n "$SAMPLER" ] || return 0
  local rc=0 t1
  t1=$(date +%s.%N)
  if kill -0 "$SAMPLER" 2>/dev/null; then
    kill -TERM "$SAMPLER" 2>/dev/null; wait "$SAMPLER" || rc=$?
  else
    wait "$SAMPLER"; rc=$?; [ "$rc" = 0 ] && rc=99; log "sampler $SAMPLER_LABEL had exited before the stage ended"
  fi
  SAMPLER=
  "$HOST_PY" "$REPO/tools/bench/validate.py" sampler --dir "$OUT" --label "$SAMPLER_LABEL" --exit "$rc" \
    --t0 "$SAMPLER_T0" --t1 "$t1" >> "$RUNS" 2>> "$LOG" || { log "INVALID sampler $SAMPLER_LABEL (runs.jsonl)"; FAILED=1; }
}
bench() {  # bench <label> <concurrency> <requests> <warmup> [extra load.py args]
  local label=$1 c=$2 n=$3 w=$4 rc=0; shift 4
  "$HOST_PY" "$REPO/tools/bench/load.py" --url "http://127.0.0.1:$PORT" --concurrency "$c" --requests "$n" \
    --warmup "$w" --inputs "$INPUTS" --bevdet-root "$BEVDET_ROOT" --out "$OUT" --label "$label" "$@" \
    > "$OUT/$label.client.txt" 2>> "$LOG" || rc=$?
  tee -a "$LOG" < "$OUT/$label.client.txt"
  "$HOST_PY" "$REPO/tools/bench/validate.py" run --dir "$OUT" --label "$label" --requests "$n" --warmup "$w" \
    --concurrency "$c" --exit "$rc" >> "$RUNS" 2>> "$LOG" || { log "INVALID run $label (runs.jsonl)"; FAILED=1; }
}
readyz_to() { curl -s --max-time 5 "http://127.0.0.1:$PORT/readyz" > "$1" || : > "$1"; }

# ------------------------------------------------------------------- stages
for stage in $STAGES; do
  log "== stage $stage"
  case "$stage" in
  cold)
    for k in $(seq 1 "$COLD_RUNS"); do
      service_up cold false 5
      echo "{\"run\": $k, \"ready_s\": $READY_S}" >> "$OUT/cold.jsonl"
      bench "cold$k-first" 1 1 0
      service_down || die "cleanup failed"
    done ;;
  levels)
    service_up levels true "$LEVELS_MAX_IN_FLIGHT"; start_sampler levels
    for c in 1 2 4 8; do bench "levels-c$c" "$c" "$LEVEL_REQUESTS" "$LEVEL_WARMUP"; done
    stop_sampler; service_down || die "cleanup failed" ;;
  overhead)
    k=0
    for timing in $(python3 "$REPO/tools/bench/overhead.py" order "$OVERHEAD_TRIALS"); do
      k=$((k + 1))
      service_up overhead "$timing" 5
      bench "overhead-t$k-timing-$timing" 1 "$OVERHEAD_REQUESTS" "$OVERHEAD_WARMUP"
      service_down || die "cleanup failed"
    done
    python3 "$REPO/tools/bench/overhead.py" analyze --dir "$OUT" --trials "$OVERHEAD_TRIALS" > "$OUT/overhead.json" \
      || { log "overhead analysis failed"; FAILED=1; } ;;
  admission)
    service_up admission true 5; start_sampler admission
    for c in $ADMISSION_LEVELS; do
      bench "admission-c$c-noretry" "$c" "$ADMISSION_REQUESTS" 0
      bench "admission-c$c-honor" "$c" "$ADMISSION_REQUESTS" 0 --honor-retry-after
    done
    sleep 2
    readyz_to "$OUT/admission-recovery-readyz.json"
    bench "admission-recovery" 1 5 0
    stop_sampler; service_down || die "cleanup failed" ;;
  soak)
    service_up soak true "$LEVELS_MAX_IN_FLIGHT"; start_sampler soak
    bench "soak-$SOAK_REQUESTS" 2 "$SOAK_REQUESTS" "$SOAK_WARMUP"
    stop_sampler
    readyz_to "$OUT/soak-readyz.json"
    rc=0
    ( cd "$BEVDET_ROOT" && PYTHONDONTWRITEBYTECODE=1 "$HOST_PY" "$GOLDEN_TOOL" compare \
        --url "http://127.0.0.1:$PORT" --golden "$GOLDEN_DIR/http_golden.json" ) \
      > "$OUT/soak-golden_compare.json" 2> "$OUT/soak-golden_compare.stderr" || rc=$?
    "$HOST_PY" "$REPO/tools/bench/validate.py" golden --label soak-golden --exit "$rc" \
      --file "$OUT/soak-golden_compare.json" --golden "$GOLDEN_DIR/http_golden.json" >> "$RUNS" 2>> "$LOG" \
      || { log "golden parity after soak: not executed or not passed (runs.jsonl)"; FAILED=1; }
    service_down || die "cleanup failed" ;;
  ab)
    k=0
    for timing in $(python3 "$REPO/tools/bench/overhead.py" order "$AB_TRIALS"); do
      k=$((k + 1))
      if [ "$timing" = true ]; then arm=b threads=$AB_THREADS; else arm=a threads=0; fi
      service_up ab "$AB_SERVER_TIMING" "$AB_MAX_IN_FLIGHT" "$threads"
      start_sampler "ab-t$k" 0.5
      p10=$(date +%s.%N); bench "ab-t$k-$arm-c1" 1 "$AB_C1_REQUESTS" "$AB_WARMUP"; p11=$(date +%s.%N)
      p80=$(date +%s.%N); bench "ab-t$k-$arm-c8" 8 "$AB_C8_REQUESTS" "$AB_WARMUP"; p81=$(date +%s.%N)
      record "{\"kind\": \"phase\", \"label\": \"ab-t$k\", \"arm\": \"$arm\", \"c1\": [$p10, $p11], \"c8\": [$p80, $p81]}"
      rc=0
      ( cd "$BEVDET_ROOT" && PYTHONDONTWRITEBYTECODE=1 "$HOST_PY" "$GOLDEN_TOOL" compare \
          --url "http://127.0.0.1:$PORT" --golden "$GOLDEN_DIR/http_golden.json" ) \
        > "$OUT/ab-t$k-golden_compare.json" 2> "$OUT/ab-t$k-golden_compare.stderr" || rc=$?
      stop_sampler
      "$HOST_PY" "$REPO/tools/bench/validate.py" golden --label "ab-t$k-golden" --exit "$rc" \
        --file "$OUT/ab-t$k-golden_compare.json" --golden "$GOLDEN_DIR/http_golden.json" >> "$RUNS" 2>> "$LOG" \
        || { log "golden parity in trial $k ($arm): not executed or not passed (runs.jsonl)"; FAILED=1; }
      service_down || die "cleanup failed"
    done
    "$HOST_PY" "$REPO/tools/bench/ab.py" analyze --dir "$OUT" --trials "$AB_TRIALS" > "$OUT/ab.json" \
      || { log "ab analysis failed"; FAILED=1; } ;;
  esac
done

"$HOST_PY" "$REPO/tools/bench/report.py" --dir "$OUT" > "$OUT/report.md" 2>> "$LOG" || { log "report failed"; FAILED=1; }
"$HOST_PY" "$REPO/tools/bench/criteria.py" --dir "$OUT" > "$OUT/criteria.json" 2>> "$LOG" || FAILED=1
"$HOST_PY" - "$OUT/criteria.json" <<'PY' | tee -a "$LOG" || FAILED=1
import json, sys
try:
    c = json.load(open(sys.argv[1]))
except (OSError, ValueError):
    sys.exit("criteria.json missing or unreadable")
for name, v in c.items():
    if isinstance(v, dict):
        print(f"{v['status'].upper():17} {'' if v.get('counted', True) else '(not counted) '}{name}")
PY
