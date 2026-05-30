#!/usr/bin/env bash
# drive_sync.sh
# Helpers to push/pull collector data to/from Google Drive via rclone.
#
# Usage in workflows (current layout: данные_парсера/{спам_номера_сырые/сайты_<x>_*,легальные_номера_сырые,итог_для_обучения/...}):
#   source scripts/drive_sync.sh
#   drive_pull_latest_archive "datasets/ru/raw/shards/${SHARD}" \
#       "данные_парсера/спам_номера_сырые/${DRIVE_DIR}" \
#       "партии/спам/шард_${SHARD}"   # 3rd arg = legacy fallback
#   ...crawler runs, writes to local path...
#   drive_push_archive "datasets/ru/raw/shards/${SHARD}" \
#       "данные_парсера/спам_номера_сырые/${DRIVE_DIR}"
#
# Replaces the previous git-commit + git-push pattern. No more pushes to the
# repo for collected data — everything goes to the user's personal Drive
# (5 TB quota). The git repo only stores code and seed config from now on.
#
# Requires: rclone in PATH; remote named `gdrive` configured via the
# RCLONE_CONFIG secret restored by .github/workflows/<wf>.yml setup step.

set -uo pipefail

# CRITICAL: rclone interprets the env var `RCLONE_CONFIG` as the *path* to
# the config file, not its contents. We accept the contents via this env
# var (set from the GitHub Actions secret), capture it into a different
# variable, and immediately unset RCLONE_CONFIG so subsequent rclone
# invocations use the default config path (~/.config/rclone/rclone.conf).
_DRIVE_RCLONE_CFG="${RCLONE_CONFIG:-}"
unset RCLONE_CONFIG

DRIVE_REMOTE="${DRIVE_REMOTE:-gdrive}"
DRIVE_ROOT="${DRIVE_ROOT:-phone-classifier}"

# drive_pull <local_dir_or_file> <remote_subpath>
# remote_subpath is relative to ${DRIVE_REMOTE}:${DRIVE_ROOT}
# Returns 0 even if remote dir is empty or missing (fresh first run).
drive_pull() {
  local local_path="$1"
  local remote_subpath="$2"
  local remote="${DRIVE_REMOTE}:${DRIVE_ROOT}/${remote_subpath}"

  if [[ -z "${local_path}" || -z "${remote_subpath}" ]]; then
    echo "drive_pull: usage: drive_pull <local_path> <remote_subpath>" >&2
    return 2
  fi

  # Both files and dirs use `rclone copy` — handles both transparently.
  # If `local_path` ends with a filename (has extension) treat as file.
  if [[ "${local_path}" == *.* && ! "${local_path}" == */ ]]; then
    local local_dir
    local_dir="$(dirname "${local_path}")"
    mkdir -p "${local_dir}"
    rclone copy --quiet "${remote}" "${local_dir}/" 2>&1 | tail -3 || \
      echo "drive_pull: remote ${remote} empty or missing, fresh start"
  else
    mkdir -p "${local_path}"
    rclone copy --quiet "${remote}/" "${local_path}/" 2>&1 | tail -3 || \
      echo "drive_pull: remote ${remote} empty or missing, fresh start"
  fi
  return 0
}

# drive_push <local_dir_or_file> <remote_subpath>
# Uploads to ${DRIVE_REMOTE}:${DRIVE_ROOT}/<remote_subpath>
# Retries up to 5 times with backoff on transient errors.
drive_push() {
  local local_path="$1"
  local remote_subpath="$2"
  local remote="${DRIVE_REMOTE}:${DRIVE_ROOT}/${remote_subpath}"

  if [[ -z "${local_path}" || -z "${remote_subpath}" ]]; then
    echo "drive_push: usage: drive_push <local_path> <remote_subpath>" >&2
    return 2
  fi

  if [[ ! -e "${local_path}" ]]; then
    echo "drive_push: local path ${local_path} does not exist, skip"
    return 0
  fi

  local attempt rc
  for attempt in 1 2 3 4 5; do
    if [[ -d "${local_path}" ]]; then
      rclone copy --quiet "${local_path}/" "${remote}/" 2>&1 | tail -5
      rc=${PIPESTATUS[0]}
    else
      rclone copy --quiet "${local_path}" "${remote}/" 2>&1 | tail -5
      rc=${PIPESTATUS[0]}
    fi
    if [[ "${rc}" == "0" ]]; then
      echo "drive_push: ${local_path} -> ${remote}/  ok (attempt ${attempt})"
      return 0
    fi
    echo "::warning::drive_push attempt ${attempt} failed (rc=${rc}), retrying..."
    sleep $((10 * attempt))
  done
  echo "::error::drive_push: all 5 attempts to ${remote} failed"
  return 1
}

# drive_push_archive <local_dir> <remote_subpath>
# Tar+gzips local_dir into a single batch_<UTC_TS>.tar.gz and uploads
# it to ${DRIVE_REMOTE}:${DRIVE_ROOT}/<remote_subpath>/. Each batch
# becomes one file in Drive instead of N individual files — keeps the
# Drive activity timeline tidy and uploads atomically.
#
# `local_dir` must be a directory; for a single-file payload, stage the
# file into a temp dir first and pass that.
drive_push_archive() {
  local local_dir="$1"
  local remote_subpath="$2"
  local remote="${DRIVE_REMOTE}:${DRIVE_ROOT}/${remote_subpath}"

  if [[ -z "${local_dir}" || -z "${remote_subpath}" ]]; then
    echo "drive_push_archive: usage: drive_push_archive <local_dir> <remote_subpath>" >&2
    return 2
  fi
  if [[ ! -d "${local_dir}" ]]; then
    echo "drive_push_archive: ${local_dir} is not a directory, skip"
    return 0
  fi
  if [[ -z "$(ls -A "${local_dir}" 2>/dev/null)" ]]; then
    echo "drive_push_archive: ${local_dir} is empty, skip"
    return 0
  fi

  local ts archive_name tmp_dir archive_path attempt rc
  ts="$(date -u +%Y%m%d_%H%M%S)"
  archive_name="batch_${ts}.tar.gz"
  tmp_dir="$(mktemp -d)"
  archive_path="${tmp_dir}/${archive_name}"

  # Tar the *contents* of local_dir (not the dir itself) so extraction
  # into a fresh dir reproduces the original layout.
  if ! ( cd "${local_dir}" && tar -czf "${archive_path}" . ) 2>&1; then
    echo "::error::drive_push_archive: tar failed for ${local_dir}"
    rm -rf "${tmp_dir}"
    return 1
  fi
  local sz
  sz="$(stat -c %s "${archive_path}" 2>/dev/null || echo '?')"
  echo "drive_push_archive: created ${archive_name} (${sz} bytes)"

  for attempt in 1 2 3 4 5; do
    rclone copy --quiet "${archive_path}" "${remote}/" 2>&1 | tail -5
    rc=${PIPESTATUS[0]}
    if [[ "${rc}" == "0" ]]; then
      echo "drive_push_archive: ${archive_name} -> ${remote}/  ok (attempt ${attempt})"
      rm -rf "${tmp_dir}"
      return 0
    fi
    echo "::warning::drive_push_archive attempt ${attempt} failed (rc=${rc}), retrying..."
    sleep $((10 * attempt))
  done
  echo "::error::drive_push_archive: all 5 attempts to ${remote} failed"
  rm -rf "${tmp_dir}"
  return 1
}

# drive_pull_latest_archive <local_dir> <remote_subpath> [legacy_subpath_1] [legacy_subpath_2] ...
# Looks for batch_*.tar.gz under remote_subpath, picks the lexicographically
# latest (== most recent UTC timestamp), downloads and extracts it into
# local_dir. Returns 0 even if no archives exist (fresh start).
#
# Lookup chain when remote_subpath has no archives:
#   1. Try each legacy_subpath_N in order — first one with archives wins.
#      Each represents a previous folder-layout era from prior PRs:
#         3rd arg = current_layout_minus_1
#         4th arg = current_layout_minus_2
#         ...
#      This matters during cutover periods when in-flight collector waves
#      still write to older paths while the new code looks at the new path.
#   2. Final fallback: copy any raw individual files left at the LAST
#      legacy_subpath given (or remote_subpath if none) via drive_pull,
#      so data uploaded by the pre-archive drive_push pattern isn't
#      ignored after the cutover.
drive_pull_latest_archive() {
  local local_path="$1"
  local remote_subpath="$2"
  shift 2
  local legacy_subpaths=("$@")
  local remote="${DRIVE_REMOTE}:${DRIVE_ROOT}/${remote_subpath}"

  if [[ -z "${local_path}" || -z "${remote_subpath}" ]]; then
    echo "drive_pull_latest_archive: usage: <local_dir> <remote_subpath> [legacy_subpath...]" >&2
    return 2
  fi
  mkdir -p "${local_path}"

  # --files-only is critical: without it, rclone lsf surfaces leftover
  # directories from the old drive_push nested-path bug (e.g. a stray
  # "legitimate_numbers.csv/" folder in allow/) which then trip up tar
  # with "Cannot open: Not a directory".
  local latest
  latest="$(rclone lsf --files-only "${remote}/" --include 'batch_*.tar.gz' 2>/dev/null | sort | tail -1)"

  if [[ -z "${latest}" && ${#legacy_subpaths[@]} -gt 0 ]]; then
    local legacy_subpath
    for legacy_subpath in "${legacy_subpaths[@]}"; do
      [[ -z "${legacy_subpath}" ]] && continue
      local legacy_remote="${DRIVE_REMOTE}:${DRIVE_ROOT}/${legacy_subpath}"
      local legacy_latest
      legacy_latest="$(rclone lsf --files-only "${legacy_remote}/" --include 'batch_*.tar.gz' 2>/dev/null | sort | tail -1)"
      if [[ -n "${legacy_latest}" ]]; then
        echo "drive_pull_latest_archive: no archives in ${remote}, falling back to legacy ${legacy_remote}"
        remote="${legacy_remote}"
        latest="${legacy_latest}"
        break
      fi
    done
  fi

  if [[ -z "${latest}" ]]; then
    # Final fallback: copy raw individual files (legacy drive_push pattern).
    # Use the LAST legacy_subpath as the raw-file location since that's the
    # oldest layout era and most likely to still have unarchived files left
    # over from before the drive_push_archive migration.
    local raw_subpath
    if [[ ${#legacy_subpaths[@]} -gt 0 ]]; then
      raw_subpath="${legacy_subpaths[-1]}"
    else
      raw_subpath="${remote_subpath}"
    fi
    echo "drive_pull_latest_archive: no archives anywhere, trying raw files at ${raw_subpath}"
    drive_pull "${local_path}" "${raw_subpath}"
    return 0
  fi

  local tmp_dir
  tmp_dir="$(mktemp -d)"
  if rclone copy --quiet "${remote}/${latest}" "${tmp_dir}/" 2>&1 | tail -3; then
    if ( cd "${local_path}" && tar -xzf "${tmp_dir}/${latest}" ); then
      echo "drive_pull_latest_archive: extracted ${latest} -> ${local_path}"
    else
      echo "::warning::drive_pull_latest_archive: extract failed for ${latest}"
    fi
  else
    echo "::warning::drive_pull_latest_archive: download of ${latest} failed"
  fi
  rm -rf "${tmp_dir}"
  return 0
}

# Convenience: setup rclone config from RCLONE_CONFIG env var.
# Idempotent — safe to call multiple times.
drive_setup() {
  if [[ -z "${_DRIVE_RCLONE_CFG:-}" ]]; then
    echo "::error::drive_setup: RCLONE_CONFIG env var is empty (secret not set?)"
    return 1
  fi
  mkdir -p "${HOME}/.config/rclone"
  printf '%s\n' "${_DRIVE_RCLONE_CFG}" > "${HOME}/.config/rclone/rclone.conf"
  chmod 600 "${HOME}/.config/rclone/rclone.conf"
  if ! command -v rclone >/dev/null 2>&1; then
    echo "drive_setup: installing rclone"
    curl -fsSL https://rclone.org/install.sh | sudo bash >/dev/null 2>&1
  fi
  rclone --version | head -1
  rclone lsd "${DRIVE_REMOTE}:" >/dev/null 2>&1 || {
    echo "::error::drive_setup: rclone cannot list ${DRIVE_REMOTE}: — bad config?"
    return 1
  }
  echo "drive_setup: ok, remote=${DRIVE_REMOTE} root=${DRIVE_ROOT}"
}
