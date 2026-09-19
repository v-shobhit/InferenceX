#!/usr/bin/env bash
set -eo pipefail

# DeepSeek-V4.1-Flash AgentX on GB300 with SGLang native DSpark, following the
# cookbook's verified Blackwell TP4/EP4 low-latency cell.
# https://lmsysorg.mintlify.app/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1
#
# Same cell as the GB200 arm, with two GB300-specific differences, both of which
# exist because GB300 carries 276 GiB per GPU against GB200's 186 GiB:
#   * --mem-fraction-static stays at the GB200 arm's 0.70. Raising it to 0.80 --
#     the value that preserves GB200's absolute 56 GiB of per-GPU headroom on a
#     276 GiB part -- took a CUDA OOM during AgentX warmup at concurrency 128 on
#     an external GB300 cluster, so the headroom these buffers need is not a
#     fixed number of GiB on this part. 0.70 is the only value with evidence.
#   * KV_OFFLOADING=dram selects the SGLang hierarchical cache. The AgentX corpus
#     replays ~245k-token median prompts at ~99% intra-trajectory KV-block reuse,
#     so the achieved prefix-cache hit rate is set by how much of a trajectory's
#     ~557k tokens of unique KV stays resident. That cannot be HBM-resident at any
#     useful concurrency on any part, which is what the host tier is for.
# KV_OFFLOADING=none reproduces the GB200 cell exactly.
source "$(dirname "$0")/../../benchmark_lib.sh"
check_env_vars MODEL TP EP_SIZE CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION
check_env_vars EVAL_ONLY
export GPU_COUNT="$TP"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

# Complete/resume partial downloads instead of trusting nonempty directories.
if [[ -n "${MODEL_PATH:-}" && "$MODEL_PATH" != "$MODEL" ]]; then
    hf download "$MODEL" --local-dir "$MODEL_PATH"
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

nvidia-smi
resolve_trace_source
install_agentic_deps
mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

# Agentic warmup dispatches hundreds of large prompts at once and SGLang's
# tokenizer can leave bytes unacknowledged past AIPerf's default 30 s
# TCP_USER_TIMEOUT, so Linux aborts live localhost connections.
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
# Outlast AIPerf's pooled connections so an inter-turn idle gap cannot race
# Uvicorn's five-second keep-alive closure.
export SGLANG_TIMEOUT_KEEP_ALIVE=900

# AgentX measures the thinking-on regime, which is also the committed golden-AL
# curve. SGLang ships thinking off by default for this model.
export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV41_REASONING_EFFORT=high

# One shared host copy of the two fp8 Engram tables instead of a row-sharded
# copy per rank: the SGLang analogue of the vLLM arm's Engram CPU offload. It
# frees ~46 GiB of HBM per GPU for the 1M-context prefill working set and the
# KV pool, and output is bitwise unchanged (cookbook). The first sweep ran
# with the tables on GPU and the server died on the first long AgentX prompts
# (run 35304536578: c2 came up, then the server exited on the first warmup prompt).
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1

CACHE_ARGS=()
WARMUP_ARGS=()
if require_agentic_kv_offload_backend hicache; then
    # DSv4-family HiCache rejects --hicache-size; capacity is a host/device token
    # ratio, and host bytes scale with the ratio AND mem-fraction-static, so the
    # ratio has to be sized from the node rather than copied from another arm.
    # This node: 4 x 276 GiB HBM, less ~475 GiB of FP4 weights, at 0.70 static
    # leaves roughly 300 GiB (~322 GB) of device KV, against the ~721 GB this
    # arm's dram-utilization 0.80 exposes. 721/322 = 2.2 is the ceiling, so 1.5
    # keeps a margin, and the page cache holding the checkpoint eats into free
    # host memory besides -- an external GB300 run at ratio 1.5 failed its host
    # pool with 868 GB available but only 30 GB free -- and 1.5 is a ratio this codebase
    # already ships elsewhere, so fractional ratios are supported. The B300 DSv4
    # arm's 8 is sized for a 2,964 GB node, and its own comment records the paged
    # pool failing to allocate there at ratio 4; copying it here would reproduce
    # that failure.
    HICACHE_RATIO=1.5
    CACHE_ARGS=(
        --enable-hierarchical-cache
        --hicache-ratio "$HICACHE_RATIO"
        --hicache-write-policy write_back
        --hicache-io-backend direct
        --hicache-mem-layout page_first_direct
    )
    # AIPerf owns the AgentX warmup; SGLang's per-DP warmup can time out after the
    # API is already healthy.
    WARMUP_ARGS=(--skip-server-warmup)
    echo "HiCache CPU tier: ratio=$HICACHE_RATIO, capacity=${TOTAL_CPU_DRAM_GB} GB, write_policy=write_back, io_backend=direct, mem_layout=page_first_direct"
fi

# AgentX concurrency counts live session trees, not individual requests.
# Allow subagent fan-out to exceed CONC without clipping request bursts, but
# never let the pool exceed the decode graph batch: a DSpark verify step for a
# batch above the captured 64 runs eagerly and allocates its attention
# workspace on the fly, which OOMed the H200 eval at 128 running requests
# (6.4 GiB allocation with 2 GiB free, run 35306704553). Batches within the
# graph tier reuse the capture-time workspace instead.
CUDA_GRAPH_MAX_BS=64
MAX_RUNNING_REQUESTS=$((2 * CONC))
if (( MAX_RUNNING_REQUESTS > CUDA_GRAPH_MAX_BS )); then
    MAX_RUNNING_REQUESTS=$CUDA_GRAPH_MAX_BS
fi

# Saturation arms carry a larger in-flight working set than the 30-minute
# default warmup drain allows.
if (( CONC >= 32 )); then
    export AGENTIC_WARMUP_GRACE_PERIOD=3600
fi

# Pyxis shares the host network; port 8888 can already belong to a host service.
select_available_server_port
export AIPERF_SERVER_URL="http://localhost:${PORT}"
export AIPERF_SERVER_METRICS_URLS="${AIPERF_SERVER_URL}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="sglang:"
echo "Using SGLang endpoint ${AIPERF_SERVER_URL}"

# DSpark is the checkpoint's own bundled draft: no EAGLE/MTP path and no
# --speculative-num-steps knob; the block size is the only tunable. Golden AL:
# golden_al_distribution/dsv41flash_dspark.yaml, thinking_on, five draft tokens.
# Throughput fixes acceptance to AL 3.51; accuracy evals keep real verification.
DSPARK_BLOCK_SIZE=5
DSV41_GOLDEN_AL=3.51
if [[ "${EVAL_ONLY}" != true ]]; then
    export SGLANG_SIMULATE_ACC_LEN="$DSV41_GOLDEN_AL"
    export SGLANG_SIMULATE_ACC_METHOD=match-expected
    export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
fi
echo "DSpark block size: $DSPARK_BLOCK_SIZE, golden AL=$DSV41_GOLDEN_AL"

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0 --port "$PORT"
    --trust-remote-code
    --tp "$TP" --ep-size "$EP_SIZE"
    # Backends resolve automatically (dsv4 / flashinfer_mxfp4 / flashinfer_cutedsl
    # on Blackwell); the cookbook warns that overriding them costs decode speed.
    # The GB200 arm's 0.70, unchanged: 0.80 OOMed during AgentX warmup at c128 on
    # an external GB300 cluster, so the extra HBM does not translate into extra
    # static fraction here.
    # The bounded prefill chunk stays: the sparse-attention indexer and DSpark
    # prefill buffers scale with the chunk times the 1M context, and the default
    # 16384 chunk exhausted HBM on the first 66k-99k-token AgentX prompts.
    --mem-fraction-static 0.70
    --chunked-prefill-size 4096
    --speculative-algorithm DSPARK
    --speculative-dspark-block-size "$DSPARK_BLOCK_SIZE"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS"
    --reasoning-parser auto
    --tool-call-parser auto
    # Draft-token forward passes under long-context agentic load block the
    # scheduler long enough to trip the 1800 s default watchdog mid-warmup.
    --watchdog-timeout 3600
    --enable-metrics
    "${CACHE_ARGS[@]}"
    "${WARMUP_ARGS[@]}"
)
write_command "$RESULT_DIR/sglang_command.txt" "${SGLANG_CMD[@]}"
{
    echo "=== SGLANG_* env vars at launch ==="
    env | grep -E '^SGLANG_' | sort
    echo "==================================="
} | tee "$SERVER_LOG"
"${SGLANG_CMD[@]}" >> "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [[ "${EVAL_ONLY}" == true ]]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --server-metrics ${AIPERF_SERVER_METRICS_URLS}"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
