.NOTPARALLEL:

test: test-rust test-python

test-rust:
	cargo test --release
	ALTIUS_ENABLE_CLIF=1 cargo test --release

test-python:
	cd crates/altius_py && ./test.sh build

clean:
	cargo clean

.PHONY: test test-rust test-python clean
