#!/bin/bash -eux

if [ "$(poetry env info -p)" = "" ]; then
  poetry install
fi

if [ ${1:-nobuild} = "build" ]; then
  if [ -z "${GITHUB_ACTIONS}" ]; then
    poetry run maturin develop -r --target-dir ./target
  else
    poetry run maturin develop -r
  fi
fi

unset GOMP_CPU_AFFINITY

poetry run python -m pytest . -n 16
