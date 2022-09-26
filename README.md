# altius

[![CI](https://github.com/maekawatoshiki/altius/workflows/CI/badge.svg)](https://github.com/maekawatoshiki/altius/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/maekawatoshiki/altius/branch/main/graph/badge.svg)](https://codecov.io/gh/maekawatoshiki/altius)

Small DNN runtime written in Rust

# Requirements

- [git-lfs](https://github.com/git-lfs/git-lfs) to pull ONNX models
- Python 3.x (Used in some tests; You can disable them by just ignoring tests in `./altius-py`)

# Run

```sh
export RUST_LOG=debug
cargo run --release --example mnist
cargo run --release --example mobilenet
cargo run --release --example mobilenet --features cuda # -- --profile
NO_AFFINITY=1 OPENBLAS_NUM_THREADS=8 cargo run --release --example mobilenet --features openblas # -- --profile
cargo run --release --example mobilenet --features accelerate # for macOS
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
