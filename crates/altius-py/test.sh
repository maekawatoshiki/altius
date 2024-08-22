#!/bin/bash -eux

if [ ! -d ".venv" ]; then
    uv sync
fi

export RUST_LOG=INFO

if [ "${1:-nobuild}" = "build" ]; then
    if [ -z "${GITHUB_ACTIONS}" ]; then
        uv run maturin develop -r --target-dir ./target > /dev/null
    else
        uv run maturin develop -r > /dev/null
    fi
fi

unset GOMP_CPU_AFFINITY

uv run python -m pytest . -n 16
