#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MAIN_PY="$PROJECT_ROOT/main.py"
OUT_DIR="${OUT_DIR:-results/lcsr_random_search}"
ABS_OUT_DIR="$PROJECT_ROOT/$OUT_DIR"
CSV_PATH="$ABS_OUT_DIR/random_search.csv"
CKPT_ROOT="$ABS_OUT_DIR/ckpts"
TOPK_PATH="$ABS_OUT_DIR/topk_checkpoints.tsv"
SEARCH_SEED="${SEARCH_SEED:-20260623}"
DEVICE="${DEVICE:-cuda:0}"
DRY_RUN="${DRY_RUN:-0}"
DATASET="ArXiv"
TRIALS="${TRIALS:-50}"
KEEP_TOP_K="${KEEP_TOP_K:-3}"
RETAIN_METRIC="${RETAIN_METRIC:-ARI}"
PYTHON_BIN="${PYTHON:-$(command -v python)}"

mkdir -p "$ABS_OUT_DIR/logs" "$CKPT_ROOT"

RANDOM="$SEARCH_SEED"

LAMBDAS=(0.002 0.005 0.008 0.01)
WARMUPS=(25 50 75)
RHOS=(0.15 0.20 0.25 0.30)
KMAXS=(2 3 4 5)
POOLS=(16 24 32 48 64)
MARGINS=(0.00 0.05 0.10 0.15 0.20)
BANK_SIZES=(32 48 64 80 96 128)
GNN_TYPES=(gcn sage)
HIDDEN_CHANNELS=(128 256 512)
NUM_LAYERS=(1 2 3)
EPOCHS=(150 200 250)
P_FMS=(0.0 0.1 0.2)
P_EDS=(0.1 0.2 0.4 0.6)

REQUIRED_BACKBONE_ARGS=(
  "--gnn_type"
  "--hidden_channels"
  "--num_layers"
  "--epochs"
  "--p_fm1"
  "--p_ed1"
  "--p_fm2"
  "--p_ed2"
)

pick_from() {
  local -n arr_ref="$1"
  echo "${arr_ref[$((RANDOM % ${#arr_ref[@]}))]}"
}

run_main() {
  local -a cmd=("$PYTHON_BIN" "$MAIN_PY" "$@")
  (
    cd "$PROJECT_ROOT" || exit 1
    PYTHONPATH="$PWD" "${cmd[@]}"
  )
}

check_cli_support() {
  local help_output
  help_output="$(run_main --model LCSR --dataset "$DATASET" -- --help 2>&1)" || true
  local missing=()
  local arg
  for arg in "${REQUIRED_BACKBONE_ARGS[@]}"; do
    if [[ "$help_output" != *"$arg"* ]]; then
      missing+=("$arg")
    fi
  done
  if (( ${#missing[@]} > 0 )); then
    echo "[LCSR SEARCH] ERROR: benchmark/LCSR/main.py on this machine does not support the new backbone search args: ${missing[*]}" >&2
    echo "[LCSR SEARCH] Sync the updated benchmark/LCSR/main.py before running this search." >&2
    exit 2
  fi
}

check_cli_support

build_command() {
  local lambda="$1"
  local warmup="$2"
  local rho="$3"
  local kmax="$4"
  local pool="$5"
  local margin="$6"
  local bank_size="$7"
  local gnn_type="$8"
  local hidden_channels="$9"
  local num_layers="${10}"
  local epochs="${11}"
  local p_fm1="${12}"
  local p_ed1="${13}"
  local p_fm2="${14}"
  local p_ed2="${15}"
  local trial_ckpt_dir="${16}"
  local -a cmd=(
    "PYTHONPATH=$PROJECT_ROOT"
    "$PYTHON_BIN" "main.py"
    --model LCSR
    --dataset "$DATASET"
    --seed 0
    --runs 5
    --device "$DEVICE"
    --ckpt_dir "$trial_ckpt_dir"
    --
    --lcsr-mode add_only
    --lcsr-support-source freq
    --lcsr-positive-mode linear
    --fcrs-extra-lambda "$lambda"
    --fcrs-extra-warmup "$warmup"
    --lcsr-rho "$rho"
    --lcsr-kmax "$kmax"
    --lcsr-candidate-pool-size "$pool"
    --gnn_type "$gnn_type"
    --hidden_channels "$hidden_channels"
    --num_layers "$num_layers"
    --epochs "$epochs"
    --p_fm1 "$p_fm1"
    --p_ed1 "$p_ed1"
    --p_fm2 "$p_fm2"
    --p_ed2 "$p_ed2"
    --lcsr-margin "$margin"
    --lcsr-candidate-bank-size "$bank_size"
    --lcsr-batch-local-semantics candidate_bank_v2
    --lcsr-force-batch-local
    --lcsr-csv-path "$CSV_PATH"
  )
  (
    cd "$PROJECT_ROOT" || exit 1
    printf '%q ' "${cmd[@]}"
  )
}

run_trial() {
  local trial_idx="$1"
  local cmd_str="$2"
  local param_desc="$3"
  local log_path="$4"
  local trial_seed="$5"

  {
    echo "[LCSR SEARCH] dataset=$DATASET"
    echo "[LCSR SEARCH] trial_index=$trial_idx/$TRIALS"
    echo "[LCSR SEARCH] random_seed=$trial_seed"
    echo "[LCSR SEARCH] command=$cmd_str"
    echo "[LCSR SEARCH] params=$param_desc"
    echo "[LCSR SEARCH] csv_path=$CSV_PATH"
    echo "[LCSR SEARCH] required_path=candidate_bank_v2"
  } | tee "$log_path"

  eval "$cmd_str" >>"$log_path" 2>&1
}

extract_metric_from_log() {
  local log_path="$1"
  local metric_name="$2"
  local metrics_line
  metrics_line="$(grep "Compact Results:" -A1 "$log_path" | tail -n1)"

  if [[ "$metric_name" == "MEAN4" || "$metric_name" == "AVG4" ]]; then
    local mean4_values
    mean4_values="$(echo "$metrics_line" | sed -nE 's/.*NMI=([0-9.+-]+).*, ARI=([0-9.+-]+).*, ACC=([0-9.+-]+).*, F1=([0-9.+-]+).*/\1 \2 \3 \4/p')"
    if [[ -z "$mean4_values" ]]; then
      return 0
    fi
    echo "$mean4_values" | awk '{printf "%.6f", ($1 + $2 + $3 + $4) / 4}'
    return 0
  fi

  echo "$metrics_line" | sed -nE "s/.*${metric_name}=([0-9.+-]+).*/\\1/p"
}

extract_core_metrics_from_log() {
  local log_path="$1"
  local metrics_line
  metrics_line="$(grep "Compact Results:" -A1 "$log_path" | tail -n1)"
  echo "$metrics_line" | sed -nE 's/.*NMI=([0-9.+-]+).*, ARI=([0-9.+-]+).*, ACC=([0-9.+-]+).*, F1=([0-9.+-]+).*/\1\t\2\t\3\t\4/p'
}

update_topk_retention() {
  local trial_name="$1"
  local score="$2"
  local ckpt_path="$3"
  local log_path="$4"
  local param_desc="$5"
  local metrics_tuple="$6"

  if [[ -z "$score" ]]; then
    echo "[LCSR SEARCH] WARNING: could not parse $RETAIN_METRIC for $trial_name; keeping checkpoint at $ckpt_path" | tee -a "$log_path"
    return
  fi

  local nmi_score=""
  local ari_score=""
  local acc_score=""
  local f1_score=""
  if [[ -n "$metrics_tuple" ]]; then
    IFS=$'\t' read -r nmi_score ari_score acc_score f1_score <<<"$metrics_tuple"
  fi

  {
    printf 'score_metric\tscore\tNMI\tARI\tACC\tF1\ttrial\tckpt_path\tlog_path\tparams\n'
    if [[ -f "$TOPK_PATH" ]]; then
      tail -n +2 "$TOPK_PATH"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$RETAIN_METRIC" "$score" "$nmi_score" "$ari_score" "$acc_score" "$f1_score" "$trial_name" "$ckpt_path" "$log_path" "$param_desc"
  } >"$TOPK_PATH.unsorted"

  {
    head -n 1 "$TOPK_PATH.unsorted"
    tail -n +2 "$TOPK_PATH.unsorted" | sort -t $'\t' -k2,2gr | awk -F '\t' '!seen[$7]++' | head -n "$KEEP_TOP_K"
  } >"$TOPK_PATH"
  rm -f "$TOPK_PATH.unsorted"

  while IFS=$'\t' read -r kept_metric kept_score kept_nmi kept_ari kept_acc kept_f1 kept_trial kept_ckpt kept_log kept_params; do
    [[ -n "$kept_trial" ]] && kept_trials["$kept_trial"]=1
  done < <(tail -n +2 "$TOPK_PATH")

  for ckpt_dir in "$CKPT_ROOT"/*; do
    [[ -d "$ckpt_dir" ]] || continue
    local ckpt_trial
    ckpt_trial="$(basename "$ckpt_dir")"
    if [[ -z "${kept_trials[$ckpt_trial]:-}" ]]; then
      rm -rf "$ckpt_dir"
    fi
  done

  echo "[LCSR SEARCH] top-$KEEP_TOP_K $RETAIN_METRIC checkpoints updated:" | tee -a "$log_path"
  cat "$TOPK_PATH" | tee -a "$log_path"
}

dry_run_printed=0
failed=0

for trial in $(seq 1 "$TRIALS"); do
  trial_name=$(printf "arxiv_%03d" "$trial")
  log_path="$ABS_OUT_DIR/logs/${trial_name}.log"
  trial_ckpt_dir="$CKPT_ROOT/${trial_name}"
  trial_seed=$((SEARCH_SEED + trial - 1))

  if [[ "$trial" -eq 1 ]]; then
    lambda=0.005
    warmup=50
    rho=0.20
    kmax=3
    pool=32
    margin=0.15
    bank_size=64
    gnn_type=sage
    hidden_channels=256
    num_layers=3
    epochs=200
    p_fm1=0.0
    p_ed1=0.4
    p_fm2=0.0
    p_ed2=0.4
  else
    while true; do
      lambda="$(pick_from LAMBDAS)"
      warmup="$(pick_from WARMUPS)"
      rho="$(pick_from RHOS)"
      kmax="$(pick_from KMAXS)"
      pool="$(pick_from POOLS)"
      margin="$(pick_from MARGINS)"
      bank_size="$(pick_from BANK_SIZES)"
      gnn_type="$(pick_from GNN_TYPES)"
      hidden_channels="$(pick_from HIDDEN_CHANNELS)"
      num_layers="$(pick_from NUM_LAYERS)"
      epochs="$(pick_from EPOCHS)"
      p_fm1="$(pick_from P_FMS)"
      p_ed1="$(pick_from P_EDS)"
      p_fm2="$(pick_from P_FMS)"
      p_ed2="$(pick_from P_EDS)"
      if (( pool >= kmax && bank_size >= pool )); then
        break
      fi
    done
  fi

  param_desc="lambda=$lambda warmup=$warmup rho=$rho kmax=$kmax pool=$pool margin=$margin bank_size=$bank_size gnn_type=$gnn_type hidden_channels=$hidden_channels num_layers=$num_layers epochs=$epochs p_fm1=$p_fm1 p_ed1=$p_ed1 p_fm2=$p_fm2 p_ed2=$p_ed2 batch_local_semantics=candidate_bank_v2 force_batch_local=true"
  cmd_str="$(build_command "$lambda" "$warmup" "$rho" "$kmax" "$pool" "$margin" "$bank_size" "$gnn_type" "$hidden_channels" "$num_layers" "$epochs" "$p_fm1" "$p_ed1" "$p_fm2" "$p_ed2" "$trial_ckpt_dir")"

  if [[ "$DRY_RUN" == "1" ]]; then
    if (( dry_run_printed < 5 )); then
      echo "[LCSR SEARCH][DRY_RUN] dataset=$DATASET trial_index=$trial/$TRIALS random_seed=$trial_seed"
      echo "[LCSR SEARCH][DRY_RUN] command=$cmd_str"
      echo "[LCSR SEARCH][DRY_RUN] params=$param_desc"
      echo "[LCSR SEARCH][DRY_RUN] csv_path=$CSV_PATH"
    fi
    dry_run_printed=$((dry_run_printed + 1))
    continue
  fi

  if ! run_trial "$trial" "$cmd_str" "$param_desc" "$log_path" "$trial_seed"; then
    echo "[LCSR SEARCH] Trial failed: dataset=$DATASET trial=$trial log=$log_path" | tee -a "$log_path"
    failed=$((failed + 1))
    rm -rf "$trial_ckpt_dir"
  else
    declare -A kept_trials=()
    trial_score="$(extract_metric_from_log "$log_path" "$RETAIN_METRIC")"
    trial_metrics="$(extract_core_metrics_from_log "$log_path")"
    update_topk_retention "$trial_name" "$trial_score" "$trial_ckpt_dir" "$log_path" "$param_desc" "$trial_metrics"
  fi
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[LCSR SEARCH][DRY_RUN] Printed first 5 commands for $DATASET."
  exit 0
fi

if (( failed > 0 )); then
  echo "[LCSR SEARCH] dataset=$DATASET completed with $failed failed trials."
  exit 1
fi

echo "[LCSR SEARCH] dataset=$DATASET completed successfully."
