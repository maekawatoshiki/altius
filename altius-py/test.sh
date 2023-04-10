#!/bin/bash -eux

VENVDIR=.env

if [ ! -e $VENVDIR ]; then
  python -m venv $VENVDIR
  source $VENVDIR/bin/activate
  python -m pip install -r ./requirements.txt
else
  source $VENVDIR/bin/activate
fi

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

if [ ${1:-nobuild} = "build" ]; then
  if [ -z "${GITHUB_ACTIONS}" ]; then
    maturin develop -r --target-dir ./target --features "${ALTIUSPY_FEATURES:-blis}"
  else
    maturin develop -r --features "${ALTIUSPY_FEATURES:-blis}"
  fi
fi

python -m pytest . -n 8
