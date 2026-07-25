#!/usr/bin/env bash

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
