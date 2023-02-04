#!/bin/bash -eux

export OMP_PLACES=cores

EXAMPLE=${1:-mobilenet}

cargo run --release --example $EXAMPLE -- --profile ${@:2}
