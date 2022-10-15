#!/bin/bash -eux

EXAMPLE=${1:-mobilenet}
NPROC=1

if [ $EXAMPLE = "vit" ]; then
  NPROC=$(nproc)
  # NPROC=32
  # NPROC=8
  printf '\x1B[33mAdjust $NPROC in this script for your environment :)\x1B[0m\n'
  sleep 3
fi

# export OMP_PROC_BIND=TRUE
export GOMP_CPU_AFFINITY="0-$(expr $NPROC - 1)"
export BLIS_NUM_THREADS=$NPROC
export RUST_LOG=debug
export RUSTFLAGS="-C target-cpu=native"

# cargo r -r --features cuda --example $EXAMPLE -- --profile
cargo r -r --features blis --example $EXAMPLE -- --profile
