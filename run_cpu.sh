#!/bin/bash -eux

export OMP_PLACES=cores
export GOMP_CPU_AFFINITY="0-7"

EXAMPLE=${1:-mobilenet_cpu}

cargo run --features cpu-backend --example $EXAMPLE -- ${@:2}
