# altius

[![CI](https://github.com/maekawatoshiki/altius/workflows/CI/badge.svg)](https://github.com/maekawatoshiki/altius/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/maekawatoshiki/altius/branch/main/graph/badge.svg)](https://codecov.io/gh/maekawatoshiki/altius)

Small DNN runtime written in Rust

Feel free to create [issues](https://github.com/maekawatoshiki/altius/issues) and [discussions](https://github.com/maekawatoshiki/altius/discussions)

# Requirements

- Python 3.x (Used in some tests; You can disable them by just ignoring tests in `./altius-py`)

# Run

```sh
# Download large models.
(cd models && ./download.sh)

# Run examples.
# {mnist, mobilenet, vit} are available.
./run.sh mnist
./run.sh mobilenet
./run.sh vit # You can specify the number of threads in ./session/examples/vit.rs

# For macOS, you can use 'accelerate' library.
RUSTFLAGS="-C target-cpu=native" \
    cargo run --release --example mobilenet --features accelerate
```

# Run from WebAssembly

Currently, mobilenet v3 runs on web browsers.

```sh
cd wasm
cargo install wasm-pack
wasm-pack build --target web
python -m http.server 8080 
```

# Run from Python

```sh
cd altius-py
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
maturin develop -r
python mobilenet.py
```
