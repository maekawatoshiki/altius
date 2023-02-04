#!/bin/sh -eux

export OMP_PLACES=cores

EXAMPLE=${1:-mobilenet}

cargo bench --features blis
