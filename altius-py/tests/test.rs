#[test]
fn run_test() {
    std::process::Command::new("bash")
        .arg("./test.sh")
        .spawn()
        .unwrap()
        .wait()
        .unwrap();
}
