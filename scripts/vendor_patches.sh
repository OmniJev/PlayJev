#!/usr/bin/env bash
# For every vendored game that still carries the upstream .git (shallow clone), record what we changed relative to
# upstream as games/<id>/vendor.patch plus the upstream commit in games/<id>/UPSTREAM, then drop the nested .git so
# the PlayJev repo can track the files itself. Idempotent; run before the first top-level commit.
set -euo pipefail
cd "$(dirname "$0")/../games"
for d in */; do
  d=${d%/}; [ -d "$d/.git" ] || continue
  url=$(git -C "$d" remote get-url origin 2>/dev/null || echo unknown)
  sha=$(git -C "$d" rev-parse HEAD)
  printf 'upstream: %s\ncommit: %s\nvendored: %s\n' "$url" "$sha" "$(date -u +%F)" > "$d/UPSTREAM"
  # tracked-file modifications only; our new files (pj_*.js, NOTES.md, levels/...) are additions, not patches
  if git -C "$d" diff --quiet; then rm -f "$d/vendor.patch"; echo "$d: no vendor patch"; else git -C "$d" diff > "$d/vendor.patch"; echo "$d: vendor.patch $(wc -l < "$d/vendor.patch") lines"; fi
  if [ "${DROP_GIT:-0}" = 1 ]; then rm -rf "$d/.git"; fi
done
