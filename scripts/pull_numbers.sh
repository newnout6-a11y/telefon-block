#!/usr/bin/env bash
# pull_numbers.sh
# One-command pull of the latest consolidated phone-number tables
# from Google Drive to a local directory.
#
# Usage:
#   bash scripts/pull_numbers.sh                     # default → ./numbers/
#   bash scripts/pull_numbers.sh ~/Downloads/numbers
#
# Prerequisites:
#   * rclone installed locally (https://rclone.org/install/)
#   * gdrive remote configured the same way the GitHub Actions runner does
#     (rclone config create gdrive drive scope=drive). The remote name
#     can be overridden via DRIVE_REMOTE; the project root inside Drive
#     via DRIVE_ROOT.
#
# What it pulls:
#   gdrive:phone-classifier/данные_парсера/итог_для_обучения/все_номера_с_метками/batch_<latest>.tar.gz
#   contains every consolidated table the pipeline produces:
#       * ru_reputation_raw.csv          (every block-side number ever seen)
#       * ru_reputation_evidence.csv     (per-source evidence rows)
#       * legitimate_numbers.csv         (allow-side numbers)
#       * ru_numbers_labeled.csv         (block ∪ allow with final label)
#       * ru_reputation_features.csv     (features for the model)
#
# Exit codes:
#   0  success
#   1  rclone not installed
#   2  no master archive found on Drive
#   3  download / extract failure

set -euo pipefail

OUT_DIR="${1:-./numbers}"
DRIVE_REMOTE="${DRIVE_REMOTE:-gdrive}"
DRIVE_ROOT="${DRIVE_ROOT:-phone-classifier}"
REMOTE_PATH="${DRIVE_REMOTE}:${DRIVE_ROOT}/данные_парсера/итог_для_обучения/все_номера_с_метками"

if ! command -v rclone >/dev/null 2>&1; then
  echo "ERROR: rclone is not installed. Get it from https://rclone.org/install/" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
echo "→ looking up latest master archive in ${REMOTE_PATH}/"
LATEST="$(rclone lsf --files-only "${REMOTE_PATH}/" --include 'batch_*.tar.gz' \
            2>/dev/null | sort | tail -1)"
if [[ -z "${LATEST}" ]]; then
  echo "ERROR: no batch_*.tar.gz archives in ${REMOTE_PATH}/" >&2
  echo "       (has the consolidate-assets workflow run yet?)" >&2
  exit 2
fi

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

echo "→ downloading ${LATEST} ($(rclone size --json "${REMOTE_PATH}/${LATEST}" \
        2>/dev/null | grep -oE '"bytes":[0-9]+' | cut -d: -f2 || echo '?') bytes)"
rclone copy --progress "${REMOTE_PATH}/${LATEST}" "${TMP}/" >&2
[[ -f "${TMP}/${LATEST}" ]] || { echo "ERROR: download missing"; exit 3; }

echo "→ extracting into ${OUT_DIR}/"
tar -xzf "${TMP}/${LATEST}" -C "${OUT_DIR}"

echo
echo "Pulled ${LATEST} → ${OUT_DIR}/"
( cd "${OUT_DIR}" && for f in *.csv; do
    [[ -f "$f" ]] || continue
    rows=$(wc -l < "$f")
    printf '  %-40s %s rows\n' "$f" "$rows"
done )
