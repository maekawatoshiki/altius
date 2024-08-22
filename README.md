<div align=center>
    <h1>Altius</h1>
    <a href="https://github.com/maekawatoshiki/altius/actions/workflows/ci.yml" target="_blank">
        <img alt="CI" src="https://img.shields.io/github/actions/workflow/status/maekawatoshiki/altius/ci.yml?branch=main&style=for-the-badge">
    </a>
    <a href="https://codecov.io/gh/maekawatoshiki/altius" target="_blank">
        <img alt="Coverage" src="https://img.shields.io/codecov/c/gh/maekawatoshiki/altius?style=for-the-badge">
    </a>
    <br />
    Small ONNX inference runtime written in Rust.
    <br />
    Feel free to create
    <a href="https://github.com/maekawatoshiki/altius/issues" target="_blank">
        issues
    </a>
    and 
    <a href="https://github.com/maekawatoshiki/altius/discussions" target="_blank">
        discussions!
    </a>
</div>

# Requirements

- cargo
- uv

# Run

```sh
# Download models.
(cd models && ./download.sh)
# Download minimum models.
# (cd models && ./download.sh CI)

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
cd ./crates/altius-py
uv sync
uv run maturin develop -r
uv run python mobilenet.py
```
