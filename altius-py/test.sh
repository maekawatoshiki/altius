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

maturin develop -r --target-dir ./target

python -m pytest . -n 8
