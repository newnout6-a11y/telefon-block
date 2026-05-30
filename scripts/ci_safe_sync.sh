#!/usr/bin/env bash
# Safe wrapper around `git pull --rebase --autostash` for the CI loops that
# auto-commit scraped data on every iteration.
#
# Why this exists: when two crawl-keepalive runner waves overlap (cron
# `0 */5 * * *` schedules a fresh wave every 5h while the previous wave is
# still running its 5.5h loop), `git pull --rebase --autostash` can hit a
# conflict during the autostash pop. The rebase still completes against
# upstream, the autostash leaves `<<<<<<<` / `=======` / `>>>>>>>` markers
# in the working tree, the stash stays on the stash stack, and the loop
# happily `git add`s and `git commit`s the marker-bearing files. Once the
# resulting state.json fails to parse, the crawler crashes on every subsequent
# iter — but `|| true` swallows the error, `git diff --staged --quiet` is
# true, and the shard sits idle for hours burning CI minutes for nothing.
#
# `safe_pull_rebase` detects this state and recovers: aborts any in-progress
# rebase, hard-resets the requested paths to HEAD, and drops leftover stashes
# so the same conflict doesn't recur on every iter. Returns 1 when a conflict
# was cleaned up so the caller can `sleep && continue` to the next iter.
#
# Usage:
#   source scripts/ci_safe_sync.sh
#   if ! safe_pull_rebase "datasets/ru/raw/shards/${SHARD}"; then
#       sleep 20
#       continue
#   fi
#
# `staged_diff_has_conflict_markers` is a defense-in-depth check meant to be
# called right before `git commit`. It returns 0 if the staged diff contains
# any added line starting with a conflict marker — callers should reset the
# index/working tree and skip the commit when it returns 0.

safe_pull_rebase() {
    local rc=0
    git pull --rebase --autostash 2>/dev/null || rc=$?

    local rebase_merge_dir rebase_apply_dir
    rebase_merge_dir="$(git rev-parse --git-path rebase-merge 2>/dev/null)"
    rebase_apply_dir="$(git rev-parse --git-path rebase-apply 2>/dev/null)"
    if [ -n "$rebase_merge_dir" ] && [ -d "$rebase_merge_dir" ]; then
        git rebase --abort 2>/dev/null || true
        rc=1
    fi
    if [ -n "$rebase_apply_dir" ] && [ -d "$rebase_apply_dir" ]; then
        git rebase --abort 2>/dev/null || true
        rc=1
    fi

    local has_markers=0
    if [ "$#" -gt 0 ]; then
        if grep -rlE '^(<<<<<<<|=======|>>>>>>>)' "$@" 2>/dev/null \
            | grep -q .; then
            has_markers=1
        fi
    fi

    if [ "$has_markers" = "1" ] || [ "$rc" -ne 0 ]; then
        echo "::warning::safe_pull_rebase: conflict / autostash failure detected — restoring tree"
        if [ "$#" -gt 0 ]; then
            git checkout HEAD -- "$@" 2>/dev/null || true
        fi
        while git stash list 2>/dev/null | grep -q '^stash@'; do
            git stash drop 2>/dev/null || break
        done
        return 1
    fi

    return 0
}

staged_diff_has_conflict_markers() {
    git diff --staged 2>/dev/null \
        | grep -qE '^\+(<<<<<<<|=======|>>>>>>>)'
}
