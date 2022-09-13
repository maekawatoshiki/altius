# altius

[![CI](https://github.com/maekawatoshiki/altius/workflows/CI/badge.svg)](https://github.com/maekawatoshiki/altius/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/maekawatoshiki/altius/branch/main/graph/badge.svg)](https://codecov.io/gh/maekawatoshiki/altius)

Small DNN runtime written in Rust

# Requirements

- [git-lfs](https://github.com/git-lfs/git-lfs) to pull ONNX models

# Run

```sh
cargo run --release --example mnist
cargo run --release --example mobilenet
cargo run --release --example mobilenet --features cuda # -- --profile
```

# Run from Python

```sh
cd altius-py
python -m venv .env
source .env/bin/activate
pip install maturin
pip install -r requirements.txt
maturin develop -r
python test.py
```
