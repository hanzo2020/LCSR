#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MAIN_PY="$PROJECT_ROOT/main.py"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
OUT_DIR="${OUT_DIR:-results/lcsr_random_search}"
ABS_OUT_DIR="$PROJECT_ROOT/$OUT_DIR"
CSV_PATH="$ABS_OUT_DIR/random_search.csv"
SEARCH_SEED="${SEARCH_SEED:-20260623}"
DEVICE="${DEVICE:-cuda:0}"
DRY_RUN="${DRY_RUN:-0}"
DATASET="Photo"
TRIALS="${TRIALS:-200}"
PYTHON_BIN="${PYTHON:-$(command -v python)}"

mkdir -p "$ABS_OUT_DIR/logs"

RANDOM="$SEARCH_SEED"
MARGIN_ENABLED=1

if [[ "$MARGIN_ENABLED" -eq 1 ]]; then
  echo "[LCSR SEARCH] Full-batch pair-package uses --lcsr-margin on $DATASET; margin search enabled."
else
  echo "[LCSR SEARCH] Full-batch pair-package does not use --lcsr-margin on $DATASET; margin search disabled."
fi

LAMBDAS=(0.0005 0.001 0.002 0.003 0.005 0.008 0.01)
WARMUPS=(0 10 25 50 75)
RHOS=(0.10 0.15 0.20 0.25 0.30 0.35 0.40)
KMAXS=(2 3 4 5 6)
POOLS=(8 12 16 24 32)
QUANTILES=(0.05 0.075 0.10 0.15 0.20 0.25 0.30)
MARGINS=(0.00 0.05 0.10 0.15)

pick_from() {
  local -n arr_ref="$1"
  echo "${arr_ref[$((RANDOM % ${#arr_ref[@]}))]}"
}

build_command() {
  local lambda="$1"
  local warmup="$2"
  local rho="$3"
  local kmax="$4"
  local pool="$5"
  local quantile="$6"
  local margin="${7:-}"
  local -a cmd=(
    "$PYTHON_BIN" "$MAIN_PY"
    --model LCSR
    --dataset "$DATASET"
    --seed 0
    --runs 5
    --device "$DEVICE"
    --
    --lcsr-mode add_only
    --lcsr-support-source mu
    --lcsr-positive-mode quantile_hinge
    --lcsr-positive-quantile "$quantile"
    --fcrs-extra-lambda "$lambda"
    --fcrs-extra-warmup "$warmup"
    --lcsr-rho "$rho"
    --lcsr-kmax "$kmax"
    --lcsr-candidate-pool-size "$pool"
    --lcsr-csv-path "$CSV_PATH"
  )
  if [[ -n "$margin" ]]; then
    cmd+=(--lcsr-margin "$margin")
  fi
  printf '%q ' "${cmd[@]}"
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
  } | tee "$log_path"

  eval "$cmd_str" >>"$log_path" 2>&1
}

dry_run_printed=0
failed=0

for trial in $(seq 1 "$TRIALS"); do
  trial_name=$(printf "photo_%03d" "$trial")
  log_path="$ABS_OUT_DIR/logs/${trial_name}.log"
  trial_seed=$((SEARCH_SEED + trial - 1))

  if [[ "$trial" -eq 1 ]]; then
    lambda=0.003
    warmup=25
    rho=0.25
    kmax=4
    pool=16
    quantile=0.10
    margin=0.00
  else
    while true; do
      lambda="$(pick_from LAMBDAS)"
      warmup="$(pick_from WARMUPS)"
      rho="$(pick_from RHOS)"
      kmax="$(pick_from KMAXS)"
      pool="$(pick_from POOLS)"
      quantile="$(pick_from QUANTILES)"
      margin="$(pick_from MARGINS)"
      if (( pool >= kmax )); then
        break
      fi
    done
  fi

  param_desc="lambda=$lambda warmup=$warmup rho=$rho kmax=$kmax pool=$pool quantile=$quantile"
  if [[ "$MARGIN_ENABLED" -eq 1 ]]; then
    param_desc="$param_desc margin=$margin"
    cmd_str="$(build_command "$lambda" "$warmup" "$rho" "$kmax" "$pool" "$quantile" "$margin")"
  else
    cmd_str="$(build_command "$lambda" "$warmup" "$rho" "$kmax" "$pool" "$quantile")"
  fi

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
