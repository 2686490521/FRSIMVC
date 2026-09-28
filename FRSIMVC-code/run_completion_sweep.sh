#!/usr/bin/env bash
set -uo pipefail

# ============================================================
# FSRIMVC outline-v9  --  Module III: Cross-Client Semantic-Spatial Completion
#
#   ONE shared training trajectory per (dataset, seed); the eight arms
#   (B0 E0 G0 D0 C025 C050 C075 C100) are evaluated as a read-only by-pass of
#   that single trajectory:
#
#     B0    frozen F05 + BSSAT-r1, original masked fusion          (baseline)
#     E0    same snapshot, refreshed clustering head, fusion on m  (adapter)
#     G0    E0 + the missing view's own graph prediction           (coverage w)
#     D0    missing slots filled with the donor distribution q     (coverage w)
#     C*    h_comp = (1-lambda) h_graph + lambda h_sem, same w     (candidate)
#
#   Everything else is frozen: A3, beta=0.5, epsilon=0.1, alpha=0, dual,
#   balanced, global_consensus=false, reliability_adaptive_graph=false,
#   random @ rho=0.3, mask-before-SLIC, tau=0.5, 300 epochs.
#
#   Job count: datasets x seeds training tasks; each task writes 8 arm records.
#   Do NOT report the arm records as independent training runs.
#
# ------------------------------------------------------------
# OOM protection (same lesson as v7 / the 10 lost runs):
#   * GPUs probed with nvidia-smi; cards below MIN_FREE_MEM_MB are dropped;
#   * MAX_JOBS_PER_GPU=1 (one sequential worker per card);
#   * PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;
#   * a job that dies with a CUDA OOM signature is retried up to MAX_RETRIES
#     times on the freest card, with an identical run id / seed / mask;
#   * the batch is audited at the end: every (dataset, seed) without a complete
#     arm set is listed in missing_runs.txt.  Nothing is reported as complete
#     unless the files are really there.
# ============================================================

CONFIG="config.json"
DATASETS_STR="${DATASETS_STR:-Augsburg MDAS MUUFL Salinas Trento XuZhou}"
SEEDS_STR="${SEEDS_STR:-0 1 2 3 4 5 6 7 8 9}"
EPOCHS="${EPOCHS:-300}"
MISSING_RATE="${MISSING_RATE:-0.3}"
MISSING_MODE="${MISSING_MODE:-random}"
LAMBDAS_STR="${LAMBDAS_STR:-0.25 0.5 0.75 1.0}"
ARMS_STR="${ARMS_STR:-B0 E0 G0 D0 C025 C050 C075 C100}"
RUN_TAG="${RUN_TAG:-v1}"
RESUME="${RESUME:-1}"

GPUS_STR="${GPUS_STR:-0 1 2 3 4 5 6 7}"
MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-10000}"
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"
MAX_RETRIES="${MAX_RETRIES:-2}"
TAU="${TAU:-0.5}"
EPSILON="${EPSILON:-0.1}"
ALPHA="${ALPHA:-0.0}"
LAMBDA_SPATIAL="${LAMBDA_SPATIAL:-1.0}"
LAMBDA_SEMANTIC="${LAMBDA_SEMANTIC:-1.0}"
ADAPTER_SEED="${ADAPTER_SEED:-20260922}"
DIAGNOSTIC_SEED="${DIAGNOSTIC_SEED:-100000}"
DIAGNOSTIC_EPOCHS_STR="${DIAGNOSTIC_EPOCHS_STR:-100 200 300}"
COMPLETION_MIN_VALID_MASS="${COMPLETION_MIN_VALID_MASS:-0.5}"
ABLATION_VARIANT="${ABLATION_VARIANT:-A3}"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/ICMVC/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./results/completion_v1}"
DETERMINISTIC_MODE="${DETERMINISTIC_MODE:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ONLY_SMOKE="${ONLY_SMOKE:-false}"
VALIDATE_ONLY="${VALIDATE_ONLY:-false}"
DRY_RUN="${DRY_RUN:-false}"
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

read -r -a DATASETS <<< "$DATASETS_STR"
read -r -a SEEDS <<< "$SEEDS_STR"
read -r -a LAMBDAS <<< "$LAMBDAS_STR"
read -r -a ARMS <<< "$ARMS_STR"
read -r -a GPUS <<< "$GPUS_STR"
read -r -a DIAGNOSTIC_EPOCHS <<< "$DIAGNOSTIC_EPOCHS_STR"

if [[ "$DETERMINISTIC_MODE" == "true" ]]; then
    DET_FLAGS=(--deterministic-mode)
else
    DET_FLAGS=()
fi

if [[ "$ONLY_SMOKE" == "true" ]]; then
    OUTPUT_ROOT="./results/completion_v1_smoke"
    SEEDS=(0)
    EPOCHS=3
    RUN_TAG="smoke"
    DIAGNOSTIC_EPOCHS=(3)
fi

if [[ "$CONFIG" != /* ]]; then CONFIG="$SCRIPT_DIR/$CONFIG"; fi
if [[ "$OUTPUT_ROOT" != /* ]]; then OUTPUT_ROOT="$SCRIPT_DIR/$OUTPUT_ROOT"; fi
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"
RESULTS_ROOT="$(realpath -m "$SCRIPT_DIR/results")"
case "$OUTPUT_ROOT" in
    "$RESULTS_ROOT"|"$RESULTS_ROOT"/*) ;;
    *) echo "ERROR: OUTPUT_ROOT must stay under $RESULTS_ROOT: $OUTPUT_ROOT" >&2; exit 2 ;;
esac

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: python not found: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "ERROR: config not found: $CONFIG" >&2; exit 2; }

BATCH_ID="$(date '+%Y%m%d_%H%M%S_%N')_$$"
BATCH_DIR="$OUTPUT_ROOT/batch_${BATCH_ID}_${RUN_TAG}"
LOG_DIR="$BATCH_DIR/_launches/logs"

probe_gpus() {
    command -v nvidia-smi >/dev/null 2>&1 || { printf '%s\n' "${GPUS[*]}"; return; }
    local idx free keep=()
    while IFS=',' read -r idx free; do
        idx="$(echo "$idx" | tr -d ' ')"; free="$(echo "$free" | tr -d ' ')"
        [[ "$idx" =~ ^[0-9]+$ ]] || continue
        [[ " ${GPUS[*]} " == *" $idx "* ]] || continue
        if [[ "$free" =~ ^[0-9]+$ ]] && (( free >= MIN_FREE_MEM_MB )); then keep+=("$idx"); fi
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null)
    printf '%s\n' "${keep[@]:-}"
}

print_banner() {
    echo "============================================================"
    echo "  FSRIMVC outline-v9 -- Module III: Cross-Client Completion"
    echo "  Datasets : ${DATASETS[*]}"
    echo "  Seeds    : ${SEEDS[*]}   missing: ${MISSING_MODE}@${MISSING_RATE} tau=${TAU}"
    echo "  Epochs   : ${EPOCHS}   ablation=${ABLATION_VARIANT}"
    echo "  Frozen   : beta=0.5 epsilon=${EPSILON} alpha=${ALPHA} dual balanced"
    echo "             global_consensus=false reliability_adaptive_graph=false"
    echo "  Arms     : ${ARMS[*]}"
    echo "  Lambdas  : ${LAMBDAS[*]}   adapter_seed=${ADAPTER_SEED}"
    echo "  Diagnose : epochs ${DIAGNOSTIC_EPOCHS[*]}  seed=${DIAGNOSTIC_SEED}"
    echo "  GPUs     : ${SELECTED_GPUS[*]:-none} (requested ${GPUS[*]}, min free ${MIN_FREE_MEM_MB} MiB)"
    echo "  Training : $(( ${#DATASETS[@]} * ${#SEEDS[@]} )) tasks x ${#ARMS[@]} arms = $(( ${#DATASETS[@]} * ${#SEEDS[@]} * ${#ARMS[@]} )) arm records"
    echo "  Output   : $BATCH_DIR"
    echo "============================================================"
}

echo "Preparing batch directory: $BATCH_DIR"
mkdir -p "$LOG_DIR"

# ---- manifest / environment / source hashes -------------------------------
cat > "$BATCH_DIR/environment.json" <<EOF
{
  "batch_id": "$BATCH_ID",
  "run_tag": "$RUN_TAG",
  "hostname": "$(hostname)",
  "user": "$(whoami)",
  "python": "$("$PYTHON_BIN" -c 'import sys; print(sys.version.split()[0])')",
  "torch": "$("$PYTHON_BIN" -c 'import torch; print(torch.__version__)' 2>/dev/null || echo unknown)",
  "cuda_version": "$("$PYTHON_BIN" -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo unknown)",
  "cudnn": "$("$PYTHON_BIN" -c 'import torch; print(torch.backends.cudnn.version())' 2>/dev/null || echo unknown)",
  "numpy": "$("$PYTHON_BIN" -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo unknown)",
  "scipy": "$("$PYTHON_BIN" -c 'import scipy; print(scipy.__version__)' 2>/dev/null || echo unknown)",
  "pytorch_cuda_alloc_conf": "$PYTORCH_CUDA_ALLOC_CONF",
  "deterministic_mode": "$DETERMINISTIC_MODE",
  "timezone": "$(date '+%Z %z')"
}
EOF
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv \
    > "$BATCH_DIR/gpu_preflight.txt" 2>&1 || echo "nvidia-smi unavailable" > "$BATCH_DIR/gpu_preflight.txt"
{
    echo "{"
    first=1
    while read -r path; do
        hash_value="$(sha256sum "$path" | awk '{print $1}')"
        [[ $first -eq 1 ]] || echo ","
        printf '  "%s": "%s"' "${path#$SCRIPT_DIR/}" "$hash_value"
        first=0
    done < <(find "$SCRIPT_DIR" -maxdepth 2 -name '*.py' -not -path '*/results/*' | sort)
    [[ $first -eq 1 ]] || echo ""
    echo "}"
} > "$BATCH_DIR/source_sha256.json"

# ---- job list -------------------------------------------------------------
JOBS=()
for seed in "${SEEDS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        JOBS+=("$dataset|$seed")
    done
done

if [[ "$DRY_RUN" == "true" ]]; then
    SELECTED_GPUS=("${GPUS[@]}")
    print_banner
    for job in "${JOBS[@]}"; do echo "  ${job//|/ seed }"; done
    exit 0
fi

# ---- preflight: data paths + CUDA + config assertions ---------------------
mapfile -t SELECTED_GPUS < <(probe_gpus)
if (( ${#SELECTED_GPUS[@]} == 0 )); then
    echo "ERROR: no GPU has >= ${MIN_FREE_MEM_MB} MiB free." >&2
    cat "$BATCH_DIR/gpu_preflight.txt" >&2
    exit 2
fi

echo "Preflight: validating data paths and the frozen-baseline assertions..."
PREFLIGHT_OK=1
for dataset in "${DATASETS[@]}"; do
    if "$PYTHON_BIN" "$SCRIPT_DIR/run_experiments.py" \
        --config "$CONFIG" --datasets "$dataset" --gpu "${SELECTED_GPUS[0]}" \
        --epochs 1 --seed "${SEEDS[0]}" --missing-seed "${SEEDS[0]}" \
        --missing-mode "$MISSING_MODE" --missing-rate "$MISSING_RATE" \
        --superpixel-threshold "$TAU" --ablation-variant "$ABLATION_VARIANT" \
        --graph-beta 0.5 --sinkhorn-epsilon "$EPSILON" \
        --lambda-spatial "$LAMBDA_SPATIAL" --lambda-semantic "$LAMBDA_SEMANTIC" \
        --mass-uniform-alpha "$ALPHA" --bssat-dual --bssat-balanced \
        --no-reliability-adaptive-graph \
        --completion-enabled --completion-stage inference_only \
        --completion-lambdas "${LAMBDAS[@]}" --completion-arms "${ARMS[@]}" \
        --adapter-seed "$ADAPTER_SEED" --diagnostic-seed "$DIAGNOSTIC_SEED" \
        --diagnostic-epochs "${DIAGNOSTIC_EPOCHS[@]}" \
        --completion-min-valid-mass "$COMPLETION_MIN_VALID_MASS" \
        "${DET_FLAGS[@]}" \
        --validate-only > "$LOG_DIR/preflight_${dataset}.log" 2>&1; then
        echo "  [ok]   $dataset"
    else
        echo "  [FAIL] $dataset (see $LOG_DIR/preflight_${dataset}.log)"
        PREFLIGHT_OK=0
    fi
done
cp "$LOG_DIR/preflight_${DATASETS[0]}.log" "$BATCH_DIR/preflight.json.log" 2>/dev/null || true
if [[ "$PREFLIGHT_OK" != "1" ]]; then
    echo "ERROR: preflight failed; refusing to launch." >&2
    exit 2
fi
if [[ "$VALIDATE_ONLY" == "true" ]]; then
    SELECTED_GPUS=("${GPUS[@]}")
    print_banner
    exit 0
fi

cat > "$BATCH_DIR/manifest.json" <<EOF
{
  "outline": "v9_module_III_completion",
  "protocol_version": "completion_v1",
  "batch_id": "$BATCH_ID",
  "run_tag": "$RUN_TAG",
  "launcher": "run_completion_sweep.sh",
  "config": "$CONFIG",
  "datasets": "$(printf '%s ' "${DATASETS[@]}")",
  "seeds": "$(printf '%s ' "${SEEDS[@]}")",
  "epochs": $EPOCHS,
  "missing_mode": "$MISSING_MODE",
  "missing_rate": $MISSING_RATE,
  "superpixel_observed_threshold": $TAU,
  "ensure_node_coverage": true,
  "reference_view": 0,
  "ablation_variant": "$ABLATION_VARIANT",
  "graph_beta": 0.5,
  "sinkhorn_epsilon": $EPSILON,
  "mass_uniform_alpha": $ALPHA,
  "csata_dual_signature": true,
  "csata_balanced": true,
  "reliability_adaptive_graph": false,
  "completion_stage": "inference_only",
  "completion_lambdas": "$(printf '%s ' "${LAMBDAS[@]}")",
  "completion_arms": "$(printf '%s ' "${ARMS[@]}")",
  "adapter_seed": $ADAPTER_SEED,
  "diagnostic_seed": $DIAGNOSTIC_SEED,
  "diagnostic_epochs": "$(printf '%s ' "${DIAGNOSTIC_EPOCHS[@]}")",
  "completion_min_valid_mass": $COMPLETION_MIN_VALID_MASS,
  "primary_lambda": 0.5,
  "primary_arm": "C050",
  "training_tasks": $(( ${#DATASETS[@]} * ${#SEEDS[@]} )),
  "arm_records": $(( ${#DATASETS[@]} * ${#SEEDS[@]} * ${#ARMS[@]} )),
  "shared_trajectory": true,
  "selected_gpus": "$(printf '%s ' "${SELECTED_GPUS[@]}")",
  "output_root": "$OUTPUT_ROOT",
  "batch_dir": "$BATCH_DIR"
}
EOF

exec > >(tee -a "$BATCH_DIR/launch.out") 2>&1
print_banner

build_cmd() {
    local dataset="$1" seed="$2" gpu="$3"
    printf '%s\0' \
        --config "$CONFIG" --datasets "$dataset" --gpu "$gpu" \
        --epochs "$EPOCHS" --seed "$seed" --missing-seed "$seed" \
        --missing-mode "$MISSING_MODE" --missing-rate "$MISSING_RATE" \
        --superpixel-threshold "$TAU" --ablation-variant "$ABLATION_VARIANT" \
        --graph-beta 0.5 --sinkhorn-epsilon "$EPSILON" \
        --lambda-spatial "$LAMBDA_SPATIAL" --lambda-semantic "$LAMBDA_SEMANTIC" \
        --mass-uniform-alpha "$ALPHA" --bssat-dual --bssat-balanced \
        --no-reliability-adaptive-graph \
        --completion-enabled --completion-stage inference_only \
        --completion-lambdas "${LAMBDAS[@]}" --completion-arms "${ARMS[@]}" \
        --adapter-seed "$ADAPTER_SEED" --diagnostic-seed "$DIAGNOSTIC_SEED" \
        --diagnostic-epochs "${DIAGNOSTIC_EPOCHS[@]}" \
        --completion-min-valid-mass "$COMPLETION_MIN_VALID_MASS" \
        "${DET_FLAGS[@]}" \
        --run-id "seed_${seed}" --output-root "$BATCH_DIR" --continue-on-error
}

oom_looks_like_oom() {
    grep -a -q -E "CUDA out of memory|CUDA error: out of memory|RuntimeError: CUDA error" "$1"
}

freest_gpu() {
    local best="" bestfree=-1 idx free
    while IFS=',' read -r idx free; do
        idx="$(echo "$idx" | tr -d ' ')"; free="$(echo "$free" | tr -d ' ')"
        [[ " ${SELECTED_GPUS[*]} " == *" $idx "* ]] || continue
        [[ "$free" =~ ^[0-9]+$ ]] || continue
        if (( free > bestfree )); then bestfree=$free; best=$idx; fi
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null)
    printf '%s' "${best:-${SELECTED_GPUS[0]}}"
}

run_done() {
    # every arm of this (dataset, seed) produced its best-round metrics file
    local dataset="$1" seed="$2" arm
    local base="$BATCH_DIR/$dataset/seed_$seed/arms"
    [[ -f "$BATCH_DIR/$dataset/seed_$seed/completion_summary.json" ]] || return 1
    for arm in "${ARMS[@]}"; do
        [[ -f "$base/$arm/metrics.json" ]] || return 1
    done
    return 0
}

run_one() {
    local dataset="$1" seed="$2" gpu="$3"
    local log_path="$LOG_DIR/${dataset}__seed${seed}__gpu${gpu}.log"
    if [[ "$RESUME" == "1" ]] && run_done "$dataset" "$seed"; then
        echo "  [skip] $dataset seed=$seed already complete"
        return 0
    fi
    local attempt=1 current_gpu="$gpu" rc=1
    while (( attempt <= MAX_RETRIES + 1 )); do
        mapfile -d '' -t argv < <(build_cmd "$dataset" "$seed" "$current_gpu")
        echo "  [gpu$current_gpu] $dataset seed=$seed (attempt $attempt)"
        "$PYTHON_BIN" -u "$SCRIPT_DIR/run_experiments.py" "${argv[@]}" > "$log_path" 2>&1
        rc=$?
        (( rc == 0 )) && { echo "  [gpu$current_gpu] OK    $dataset seed=$seed"; return 0; }
        if oom_looks_like_oom "$log_path"; then
            local alt; alt="$(freest_gpu)"
            if [[ -n "$alt" && "$alt" != "$current_gpu" ]]; then
                echo "  [gpu$current_gpu] OOM   $dataset seed=$seed -> retry on gpu$alt"
                echo "  (retry: identical run id / seed / mask / config)" >> "$log_path"
                mkdir -p "$BATCH_DIR/$dataset/seed_$seed/failed_attempts"
                cp "$log_path" "$BATCH_DIR/$dataset/seed_$seed/failed_attempts/attempt_${attempt}_gpu${current_gpu}.log"
                current_gpu="$alt"
                attempt=$((attempt + 1))
                continue
            fi
        fi
        break
    done
    echo "  [gpu$current_gpu] FAIL  $dataset seed=$seed (exit=$rc) log=$log_path" >&2
    return 1
}

# ---- static partition: one sequential worker per GPU ---------------------
FAILED_COUNT=0
declare -a WORKER_PIDS=() WORKER_STATUS=()
for gi in "${!SELECTED_GPUS[@]}"; do
    gpu="${SELECTED_GPUS[$gi]}"
    status_file="$LOG_DIR/.worker_gpu${gpu}.status"
    : > "$status_file"
    (
        local_fail=0
        for ((ji=gi; ji<${#JOBS[@]}; ji+=${#SELECTED_GPUS[@]})); do
            IFS='|' read -r dataset seed <<< "${JOBS[$ji]}"
            run_one "$dataset" "$seed" "$gpu" || local_fail=$((local_fail + 1))
        done
        echo "$local_fail" > "$status_file"
    ) &
    WORKER_PIDS+=("$!"); WORKER_STATUS+=("$status_file")
done
for pid in "${WORKER_PIDS[@]}"; do wait "$pid" || true; done
for status_file in "${WORKER_STATUS[@]}"; do
    value="$(cat "$status_file" 2>/dev/null || echo 1)"
    FAILED_COUNT=$((FAILED_COUNT + ${value:-1}))
done

# ---- completeness audit --------------------------------------------------
: > "$BATCH_DIR/missing_runs.txt"
total=0; complete=0
for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        total=$((total + 1))
        if run_done "$dataset" "$seed"; then
            complete=$((complete + 1))
        else
            printf '%s seed_%s\n' "$dataset" "$seed" >> "$BATCH_DIR/missing_runs.txt"
        fi
    done
done

echo
echo "============================================================"
echo "  Completeness audit"
echo "  training tasks: expected=$total completed=$complete missing=$((total - complete))"
if (( total - complete > 0 )); then
    echo "  missing tasks listed in $BATCH_DIR/missing_runs.txt"
    sed -n '1,20p' "$BATCH_DIR/missing_runs.txt"
fi
echo "============================================================"

if [[ -x "$SCRIPT_DIR/report_completion.py" || -f "$SCRIPT_DIR/report_completion.py" ]]; then
    echo
    echo "  Aggregating..."
    "$PYTHON_BIN" "$SCRIPT_DIR/report_completion.py" --batch-dir "$BATCH_DIR" \
        2>&1 | tee "$BATCH_DIR/_launches/aggregation.log"
fi

python - "$BATCH_DIR" <<'PY' 2>/dev/null || true
import json, sys, pathlib
batch = pathlib.Path(sys.argv[1])
manifest = json.loads((batch / 'manifest.json').read_text())
manifest['status'] = 'completed' if not (batch / 'missing_runs.txt').stat().st_size else 'incomplete'
manifest['end_time'] = __import__('datetime').datetime.now().astimezone().isoformat()
(batch / 'manifest.json').write_text(json.dumps(manifest, indent=2))
print('  manifest status:', manifest['status'])
PY

echo
echo "  Finished : $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "  Batch    : $BATCH_DIR"
echo "  Log      : $BATCH_DIR/launch.out"
exit 0
