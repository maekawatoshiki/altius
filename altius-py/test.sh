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
  if [ -v $GITHUB_ACTIONS ]; then
    maturin develop --target-dir ./target --features blis
  else
    maturin develop --features blis
  fi
fi

python -m pytest . -n 8
