# altius

[![CI](https://github.com/maekawatoshiki/altius/workflows/CI/badge.svg)](https://github.com/maekawatoshiki/altius/actions/workflows/ci.yml)

Small DNN runtime written in Rust

# Run

```sh
cargo run --release --example mnist
cargo run --release --example mobilenet
cargo run --release --example mobilenet --features cuda # -- --profile
```
