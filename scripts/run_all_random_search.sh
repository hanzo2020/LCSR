#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_child() {
  local name="$1"
  local script_path="$2"
  if ! bash "$script_path"; then
    echo "[LCSR ALL SEARCH] ERROR: $name search failed"
    return 1
  fi
  return 0
}

cora_status=0
photo_status=0
arxiv_status=0

run_child "Cora" "$SCRIPT_DIR/random_search_cora.sh" || cora_status=$?
echo "[LCSR ALL SEARCH] Cora finished"

run_child "Photo" "$SCRIPT_DIR/random_search_photo.sh" || photo_status=$?
echo "[LCSR ALL SEARCH] Photo finished"

run_child "ArXiv" "$SCRIPT_DIR/random_search_arxiv.sh" || arxiv_status=$?
echo "[LCSR ALL SEARCH] ArXiv finished"

echo "[LCSR ALL SEARCH] All requested searches completed"

if (( cora_status != 0 || photo_status != 0 || arxiv_status != 0 )); then
  exit 1
fi
