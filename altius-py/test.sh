#!/bin/bash -eux

VENVDIR=.venv

if [ ! -e $VENVDIR ]; then
  rye sync
fi

source $VENVDIR/bin/activate

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

if [ ${1:-nobuild} = "build" ]; then
  if [ -z "${GITHUB_ACTIONS}" ]; then
    maturin develop -r --target-dir ./target --features "${ALTIUSPY_FEATURES:-blis}"
  else
    maturin develop -r --features "${ALTIUSPY_FEATURES:-blis}"
  fi
fi

python -m pytest . -n 8
