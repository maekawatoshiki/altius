#!/bin/bash -eux

if [ ! -d ".venv" ]; then
  if [ "$(uname)" = "Darwin" ]; then
    echo "Building on macOS"
    rye sync
  else
    echo "Building on Linux"
    rye sync --features linux
  fi
fi

if [ ${1:-nobuild} = "build" ]; then
  if [ -z "${GITHUB_ACTIONS}" ]; then
    rye run maturin develop -r --target-dir ./target
  else
    rye run maturin develop -r
  fi
fi

unset GOMP_CPU_AFFINITY

rye run python -m pytest . -n 16
