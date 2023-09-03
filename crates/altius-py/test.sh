#!/bin/bash -eux

if [ "$(poetry env info -p)" = "" ]; then
  poetry install
fi

source $(poetry env info -p)/bin/activate

if [ ${1:-nobuild} = "build" ]; then
  if [ -z "${GITHUB_ACTIONS}" ]; then
    maturin develop -r --target-dir ./target
  else
    maturin develop -r
  fi
fi

unset GOMP_CPU_AFFINITY

python -m pytest . -n 8
