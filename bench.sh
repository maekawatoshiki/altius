#!/bin/sh -eux

export OMP_PLACES=cores
export RUST_LOG=debug
export RUSTFLAGS="-C target-cpu=native"

EXAMPLE=${1:-mobilenet}

cargo bench --features blis
