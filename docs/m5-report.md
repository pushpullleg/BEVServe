# M5 report: performance characterization and load testing

Baselines for the verified BEVServe GPU container on one workstation. Everything here was measured
in a single full run on 2026-10-27 (run `m5-20261027-160635`; raw results are kept privately,
outside git). Method and criteria were fixed before the run ([m5-plan.md](m5-plan.md)). No
optimization was made and inference behavior was not changed.

**Outcome:** all harness, functional and load criteria passed. One statistical criterion, the
overhead of the optional Server-Timing instrumentation, was **not established**: the measured
effect is +1.12 ms (95% CI −8.11 to +10.35 ms) against a 3.14 ms limit. The run therefore exited
non-zero; the criterion and its threshold are unchanged.

## Environment and provenance

| Item | Value |
| --- | --- |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition (97,887 MiB), driver 570.211.01 |
| Host | AMD Ryzen Threadripper PRO 7975WX (32 cores, 64 threads), 503 GiB RAM, Linux 6.14.0-37-generic |
| Image | `sha256:4dfd2292b36ff8d9eed6d5e72bf560f0fd2523301bc922f624709da35063090b` (7.29 GB) |
| Image verification | M4 suite on this image (Stage A) plus an isolated fault-recovery retest; all checks passed on this image id, the Stage A exit-code query being superseded by the retest |
| Source | commit 3024183 plus the uncommitted M5 changes (`3024183-dirty`); the bevserve package inside the image is byte-identical to the benchmarked work tree (package fingerprint `7a1e98f0…`) |
| Checkpoint | BEVDet-R50-CBGS, sha256 `8b67e60d…3a13443`, 177,418,949 bytes |
| Inputs | 20 nuScenes val frames spread over the val set (6 JPEGs, about 860 KB per request), cycled |
| Client | closed loop, same host as the server, `tools/bench/load.py` |

Model outputs were checked after the load: the 10-frame HTTP golden parity (byte-identical boxes,
same model identity) passed after the soak.

## Method in brief

Closed-loop clients with C workers; warm-up requests excluded; latency is client wall time;
percentiles are nearest rank with 95% bootstrap intervals; throughput counts successful responses
only. Server stages come from the optional `Server-Timing` header (off in production). Resources
were sampled at 1 Hz from `/proc` and `nvidia-smi`, with GPU memory attributed only to the serving
process. Every client run, sampler, golden comparison and service removal has an execution record;
all 25 client runs, 3 samplers, 1 golden comparison and 18 of 18 service removals were valid.

## Results

### Latency and throughput (server timing on, max_in_flight 16)

| C | Requests | OK | Throughput (req/s) | p50 ms [95% CI] | p95 ms | p99 ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 300 | 300 | 9.23 | 107.7 [107.5, 108.1] | 112.1 | 114.1 |
| 2 | 300 | 300 | 17.95 | 110.3 [110.0, 110.7] | 116.8 | 118.4 |
| 4 | 300 | 300 | 30.35 | 128.4 [127.4, 129.6] | 144.4 | 147.9 |
| 8 | 300 | 300 | 42.23 | 185.7 [185.2, 186.5] | 195.7 | 202.0 |

Server stages, median ms:

| C | decode | preprocess | queue | GPU | of which forward | server total |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 14.2 | 74.8 | 0.1 | 12.4 | 11.3 | 104.2 |
| 2 | 16.6 | 75.3 | 0.1 | 12.8 | 11.5 | 107.7 |
| 4 | 20.1 | 85.2 | 0.1 | 15.3 | 13.6 | 125.6 |
| 8 | 23.0 | 94.4 | 36.4 | 22.7 | 19.7 | 182.9 |

Observations from these numbers:

- At C = 1, 85% of server time is CPU work (JPEG decode and preprocessing, 89 ms of 104 ms) and 12% is
  the GPU (12.4 ms including the copy of results to the host). The provisional design target
  (p95 server total <= 250 ms at C = 1) is met: 108.6 ms.
- Throughput rises 4.6x from C = 1 to C = 8 while median latency rises 1.7x. At C = 8, requests
  wait 36 ms for the single GPU slot, and decode and preprocessing slow as they compete for CPU.
- GPU utilization stayed low (mean 12%, max 38% over the levels stage). The server process used
  up to about 25 cores (sampled peak 2,463% CPU) at C = 8.

### Cold start (3 starts, timing off)

Ready (`/readyz` 200) in 4.80, 4.64 and 4.68 s after `compose up`; the first request took 155–157
ms, about 48 ms more than a warm request.

### Admission control (production max_in_flight 5)

| C | Client | OK | Rejected (503 `queue_full`) | Throughput (req/s) |
| --- | --- | --- | --- | --- |
| 8 | ignores Retry-After | 6 | 194 | 14.39 |
| 8 | honors Retry-After | 185 | 15 | 37.38 |
| 16 | ignores Retry-After | 7 | 193 | 15.23 |
| 16 | honors Retry-After | 155 | 45 | 37.03 |

Only 200 and 503 `queue_full` occurred, and every rejection carried `Retry-After: 1`. After the
saturation runs the service reported ready and served 5 of 5 requests. Admission is taken before
CPU decode and preprocessing, so with the production limit of 5, concurrency above 5 is rejected
even while the GPU is mostly idle. A client that retries immediately turns this into a rejection
storm (97% rejected); a client that honors `Retry-After` gets most of the throughput back. This is
measured behavior, not a change: the limit stays at 5.

### Soak (100 requests at C = 2, plus 20 warm-up)

100 of 100 succeeded (p50 125.6 ms, 15.78 req/s). Over the about 11 s of sampling, RSS grew 83 MiB
(limit 200), serving-process GPU memory grew 88 MiB (1,228 to 1,316 MiB; limit 100), and the
median latency of the last tenth was 1.03x the first (limit 1.25). Afterwards the service was ready
and the 10-frame golden parity passed. This is a short pilot soak: 11 resource samples cannot
distinguish allocator warm-up from a slow leak, and the GPU-memory growth is close to its limit.
The 1,000-request soak has not been run.

## Instrumentation overhead: not established

Design: 12 fresh services at C = 1 in ABBA order (A = Server-Timing off, B = on), 100 measured
requests each after 20 warm-up; per-trial client p50; adjacent A/B trials paired; mean paired
difference with a 95% Student t interval. Pass required the upper bound <= max(3 ms, 3% of the
off-arm p50) = 3.14 ms.

| Pair (off, on) | 1, 2 | 4, 3 | 5, 6 | 8, 7 | 9, 10 | 12, 11 |
| --- | --- | --- | --- | --- | --- | --- |
| p50 difference, on − off (ms) | −0.25 | +0.24 | +13.93 | +6.27 | −12.65 | −0.84 |

Mean +1.12 ms, SD 8.79 ms, 6 pairs (5 degrees of freedom, t = 2.571): 95% CI [−8.11, +10.35] ms.
The upper bound exceeds the limit while the estimate does not, so the result is
`insufficient_data`.

The 1,200 overhead requests are not 1,200 independent observations of the effect: each trial gives
one precise p50 (within-trial p95 − p50 is about 4 ms, and the first and last 20 requests of every
trial have medians within 1.5 ms), but the experimental unit is the service instance, so there are
6 independent pairs.

Why the interval is wide: per-trial p50 clusters around 99 ms (trials 3, 4, 5, 10), 108 ms
(1, 2, 7, 11, 12) and 112 ms (6, 9), with trial 8 at 101.6 ms; both arms occur in each cluster.
The response bodies show that the shift is in CPU work, not the GPU: preprocessing medians are
67.0–69.0 ms in the trials with p50 below 102 ms and 73.6–76.3 ms in the rest, decode is
18.7–19.2 ms in the two slowest trials and 14.2–15.2 ms elsewhere, while inference is 12.4–12.7 ms in
every trial of both arms. There is no trend over the stage, no order effect (slow levels occur in
both A and B positions and at the start, middle and end), and every start became ready in
4.5–4.7 s. The likely source is per-process CPU placement or frequency (which cores and cache
domain the decode and preprocessing threads run on); this is **not verified**, because the overhead
stage had no resource sampler and no CPU-affinity record.

What is being compared: timing on adds stage timers, the `Server-Timing` header, a GPU
synchronization after the input upload and CUDA events around the forward pass. The inference time
is the same in both arms (12.4–12.7 ms), so the GPU synchronization has no visible cost at this
resolution; the client-level comparison could not resolve effects below about ±9 ms.

Sample size: with the observed mean and SD unchanged, the upper bound reaches 3.14 ms only when
mean + t(0.975, n−1) × 8.79 / √n <= 3.14, i.e. at **76 pairs** (152 fresh services). With 6, 8 or
12 pairs the SD between paired trials would have to fall below 1.9, 2.4 or 3.2 ms. Running 152
service starts to obtain a pass was declined; the criterion is reported as not established.

## Limitations

- One machine, one GPU, client and server on the same host; client CPU use competes with server
  preprocessing at higher concurrency.
- Closed-loop load only (no open-loop arrival rates); the levels stage stops at C = 8 (max_in_flight
  16), so throughput beyond C = 8 is not characterized.
- The soak is a 100-request pilot; long-run memory behavior is not characterized.
- Instrumentation overhead is not established (above).
- The benchmarked image was built from an uncommitted work tree; the image's package was verified
  byte-identical to that tree, and this report is committed together with that tree.
