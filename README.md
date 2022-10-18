# altius

[![CI](https://github.com/maekawatoshiki/altius/workflows/CI/badge.svg)](https://github.com/maekawatoshiki/altius/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/maekawatoshiki/altius/branch/main/graph/badge.svg)](https://codecov.io/gh/maekawatoshiki/altius)

Small DNN runtime written in Rust

# Requirements

- [git-lfs](https://github.com/git-lfs/git-lfs) to pull ONNX models
- Python 3.x (Used in some tests; You can disable them by just ignoring tests in `./altius-py`)

# Run

- First of all, to download large models, run `cd models && ./download.sh`.
- Use `./run.sh` to run examples (e.g. `./run.sh {mnist, mobilenet, vit}`)
  - For `vit` (vision transformer) example, you can specify the number of threads for computation in `./session/examples/vit.rs`.
  - You can manually run examples by the following commands.

```sh
export RUST_LOG=debug
cargo run --release --example mnist
cargo run --release --example mobilenet
cargo run --release --example mobilenet --features cuda # -- --profile
NO_AFFINITY=1 OPENBLAS_NUM_THREADS=8 cargo run --release --example mobilenet --features openblas # -- --profile
BLIS_NUM_THREADS=8 cargo run --release --example vit --features blis
cargo run --release --example mobilenet --features accelerate # for macOS
```

- Recommend to use `blis` feature and set `GOMP_CPU_AFFINITY='0-31' BLIS_NUM_THREADS=32` (adjust the number of cores for your machine) for better performance

# Run from Python

```sh
cd altius-py
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
maturin develop -r
python mobilenet.py
```
