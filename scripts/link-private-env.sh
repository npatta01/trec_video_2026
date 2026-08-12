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

die() {
    printf 'error: %s\n' "$1" >&2
    exit 1
}

config_root="${XDG_CONFIG_HOME:-${HOME:-}/.config}"
[[ "$config_root" == /* ]] || die "XDG_CONFIG_HOME or HOME must resolve to an absolute path"
canonical_env="$config_root/trec_video_2026/gemini.env"

[[ -f "$canonical_env" && ! -L "$canonical_env" ]] \
    || die "canonical private environment file is missing or is not a regular file"

canonical_env="$(readlink -f -- "$canonical_env")"
[[ "$(stat -c '%a' -- "$canonical_env")" == 600 ]] \
    || die "canonical private environment file must have permissions 0600"
[[ "$(stat -c '%u' -- "$canonical_env")" == "$(id -u)" ]] \
    || die "canonical private environment file must be owned by the current user"

link_worktree() {
    local worktree_root=$1
    local private_env_dir
    local destination
    local resolved

    worktree_root=$(cd "$worktree_root" && pwd -P)
    private_env_dir="$worktree_root/private-env"
    destination="$private_env_dir/gemini.env"

    [[ -d "$private_env_dir" && ! -L "$private_env_dir" ]] \
        || die "tracked private-env directory is missing or is not a physical directory: $private_env_dir"

    if [[ -L "$destination" ]]; then
        resolved=$(readlink -f -- "$destination" || true)
        if [[ "$resolved" == "$canonical_env" ]]; then
            printf 'already linked: %s\n' "$destination"
            return
        fi
        die "refusing to replace conflicting private-env symlink: $destination"
    fi

    if [[ -e "$destination" ]]; then
        die "refusing to replace conflicting private-env path: $destination"
    fi

    ln -s -- "$canonical_env" "$destination"
    printf 'linked: %s\n' "$destination"
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
