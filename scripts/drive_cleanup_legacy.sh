#!/usr/bin/env bash
# drive_cleanup_legacy.sh
# One-shot cleanup of the legacy Drive folder layout (pre-PR-#6).
# After this script runs successfully, the Drive root contains ONLY the
# new batches/{block,allow,ready}/ tree.
#
# SAFE TO RUN ONLY AFTER:
#   * PR #6 (batches layout) is merged AND
#   * PR #7 (remove tg) is merged AND
#   * The pre-archive crawl-keepalive wave (sha=e0a69c1) has finished
#     (~03:06 MSK on the day of merge), AND
#   * crawl-keepalive's NEW wave has pushed at least one
#     batches/block/<shard>/batch_*.tar.gz per shard a/b/c/d (~04:00 MSK).
#
# Usage:
#   bash scripts/drive_cleanup_legacy.sh           # asks for confirmation
#   bash scripts/drive_cleanup_legacy.sh --force   # skip confirmation
#
# Customise via DRIVE_REMOTE / DRIVE_ROOT env vars (same as drive_sync.sh).

set -euo pipefail

DRIVE_REMOTE="${DRIVE_REMOTE:-gdrive}"
DRIVE_ROOT="${DRIVE_ROOT:-phone-classifier}"
FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

LEGACY_PATHS=(
  # Pre-PR-#6 layout (raw drive_push files):
  "shards"
  "allow"
  "master"
  "assets"
  "releases"
  # PR #6 - PR #8 English layout (now superseded by Cyrillic names):
  "batches"
  # PR #9 short Cyrillic layout (now superseded by descriptive names):
  "партии"
)

if ! command -v rclone >/dev/null 2>&1; then
  echo "ERROR: rclone is not installed." >&2
  exit 1
fi

echo "Drive root: ${DRIVE_REMOTE}:${DRIVE_ROOT}/"
echo "About to PERMANENTLY purge:"
for p in "${LEGACY_PATHS[@]}"; do echo "  - ${DRIVE_REMOTE}:${DRIVE_ROOT}/${p}/"; done

if (( ! FORCE )); then
  read -r -p "Type 'yes' to confirm: " ans
  [[ "${ans}" == "yes" ]] || { echo "aborted"; exit 1; }
fi

for p in "${LEGACY_PATHS[@]}"; do
  full="${DRIVE_REMOTE}:${DRIVE_ROOT}/${p}"
  if rclone lsd "${full}" >/dev/null 2>&1; then
    echo "→ purging ${full}/"
    rclone purge "${full}/"
  else
    echo "  (already gone) ${full}/"
  fi
done

echo
echo "Done. Remaining top-level folders:"
rclone lsd "${DRIVE_REMOTE}:${DRIVE_ROOT}/"
