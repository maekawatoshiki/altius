#!/bin/bash -eux

export OMP_PLACES=cores

EXAMPLE=${1:-mobilenet}

cargo run --release --features blis --example $EXAMPLE -- --profile ${@:2}
