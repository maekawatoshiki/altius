# altius

[![CI](https://github.com/maekawatoshiki/altius/workflows/CI/badge.svg)](https://github.com/maekawatoshiki/altius/actions/workflows/ci.yml)

Small DNN runtime written in Rust

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
maturin develop -r
python test.py
```