#!/bin/bash -eux

export OMP_PLACES=cores
export RUST_LOG=debug
export RUSTFLAGS="-C target-cpu=native"

EXAMPLE=${1:-mobilenet}

# cargo r -r --features cuda --example $EXAMPLE -- --profile
cargo r -r --features blis --example $EXAMPLE -- --profile
