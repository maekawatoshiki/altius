#!/bin/bash -eux

if [ ! -d ".venv" ]; then
    uv sync
fi

export RUST_LOG=INFO

if [ "${1:-nobuild}" = "build" ]; then
    uv run maturin develop -r > /dev/null
fi

unset GOMP_CPU_AFFINITY

n=$(nproc)
uv run python -m pytest . -n $((n > 16 ? 16 : n))
