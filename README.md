# altius

[![CI](https://github.com/maekawatoshiki/altius/workflows/CI/badge.svg)](https://github.com/maekawatoshiki/altius/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/maekawatoshiki/altius/branch/main/graph/badge.svg)](https://codecov.io/gh/maekawatoshiki/altius)

Small ONNX inference runtime written in Rust

Feel free to create [issues](https://github.com/maekawatoshiki/altius/issues) and [discussions](https://github.com/maekawatoshiki/altius/discussions)

# Requirements

- [rye](https://github.com/mitsuhiko/rye)
    - To use Python

# Run

```sh
# Download large models.
(cd models && ./download.sh)

# Run examples.
# {mnist, mobilenet, deit, vit} are available.
# You can specify the number of threads for computation by editing the code.
cargo run --release --example mnist
cargo run --release --example mobilenet
cargo run --release --example deit
cargo run --release --example vit

# Experimental CPU backend (that generates code in C)
cargo run --release --example mnist_cpu     -- --iters 10 
cargo run --release --example mobilenet_cpu -- --iters 10 --profile
cargo run --release --example deit_cpu      -- --iters 10 --threads 8 --profile
```

# Run from WebAssembly

Currently, mobilenet v3 runs on web browsers.

```sh
cd wasm
cargo install wasm-pack
wasm-pack build --target web
yarn
yarn serve
```

# Run from Python

```sh
cd altius-py
rye shell
rye sync
rye run maturin develop -r
rye run python mobilenet.py
```
