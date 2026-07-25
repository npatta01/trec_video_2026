#!/usr/bin/env bash

resolve_pinned_uvx() {
  local launcher
  if ! launcher="$(command -v uvx)"; then
    echo "error: uvx launcher is unavailable" >&2
    return 2
  fi
  if [[ "$launcher" != /* || ! -x "$launcher" ]]; then
    echo "error: uvx launcher must be an absolute executable" >&2
    return 2
  fi
  UVX_LAUNCHER="$launcher"

  local version_output executable version remainder
  if ! version_output="$(pinned_uvx --version)"; then
    echo "error: pinned inner uvx is unavailable" >&2
    return 2
  fi
  read -r executable version remainder <<<"$version_output"
  if [[ "$executable" != "uvx" || "$version" != "$UV_VERSION" ]]; then
    echo "error: pinned inner uvx version attestation failed" >&2
    return 2
  fi
}

pinned_uvx() {
  if [[ "$UVX_LAUNCHER" != /* || ! -x "$UVX_LAUNCHER" ]]; then
    echo "error: uvx launcher is not resolved" >&2
    return 2
  fi
  "$UVX_LAUNCHER" --from "uv==$UV_VERSION" uvx "$@"
}

hf_cli() {
  pinned_uvx --from "huggingface-hub==$HF_CLI_VERSION" hf "$@"
}

publish_verified_output() {
  local temporary_output_dir="$1"
  local output_dir="$2"

  if [[ -e "$output_dir" || -L "$output_dir" ]]; then
    echo "error: refusing to publish over existing output: $output_dir" >&2
    return 2
  fi

  mv -T --no-clobber -- "$temporary_output_dir" "$output_dir"
  if [[ -e "$temporary_output_dir" || ! -d "$output_dir" ]]; then
    echo "error: refusing to publish over existing output: $output_dir" >&2
    return 2
  fi
}
