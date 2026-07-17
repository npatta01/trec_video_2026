#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: %s [--all]\n' "${0##*/}" >&2
}

mode=current
case "${1:-}" in
    "") ;;
    --all) mode=all ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        usage
        exit 2
        ;;
esac

if (( $# > 1 )); then
    usage
    exit 2
fi

common_dir=$(git rev-parse --path-format=absolute --git-common-dir)
canonical_root=$(cd "$(dirname "$common_dir")" && pwd -P)
canonical_data="$canonical_root/data"

if [[ ! -d "$canonical_data" || -L "$canonical_data" ]]; then
    printf 'error: canonical data directory is missing or is not a physical directory: %s\n' \
        "$canonical_data" >&2
    exit 1
fi

canonical_data=$(cd "$canonical_data" && pwd -P)

exclude_file="$common_dir/info/exclude"
mkdir -p "$(dirname "$exclude_file")"
if [[ ! -f "$exclude_file" ]] || ! grep -Fqx '/data' "$exclude_file"; then
    printf '\n# Shared dataset directory and per-worktree links\n/data\n' >>"$exclude_file"
fi

link_worktree() {
    local worktree_root=$1
    local destination
    local resolved

    worktree_root=$(cd "$worktree_root" && pwd -P)
    destination="$worktree_root/data"

    if [[ "$worktree_root" == "$canonical_root" ]]; then
        printf 'canonical data directory: %s\n' "$canonical_data"
        return
    fi

    if [[ -L "$destination" ]]; then
        resolved=$(readlink -f -- "$destination" || true)
        if [[ "$resolved" == "$canonical_data" ]]; then
            printf 'already linked: %s -> %s\n' "$destination" "$canonical_data"
            return
        fi
        printf 'error: refusing to replace conflicting data symlink: %s\n' \
            "$destination" >&2
        return 1
    fi

    if [[ -e "$destination" ]]; then
        printf 'error: refusing to replace conflicting data path: %s\n' \
            "$destination" >&2
        return 1
    fi

    ln -s -- "$canonical_data" "$destination"
    printf 'linked: %s -> %s\n' "$destination" "$canonical_data"
}

if [[ "$mode" == all ]]; then
    while IFS= read -r record; do
        case "$record" in
            "worktree "*) link_worktree "${record#worktree }" ;;
        esac
    done < <(git worktree list --porcelain)
else
    current_root=$(git rev-parse --path-format=absolute --show-toplevel)
    link_worktree "$current_root"
fi
