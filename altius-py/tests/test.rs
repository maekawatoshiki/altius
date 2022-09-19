use std::{
    env,
    ffi::OsString,
    fs::read_dir,
    io,
    path::{Path, PathBuf},
};

use cargo_util::paths::mtime_recursive;

#[test]
fn run_test() {
    // If build artifacts are modified, run `maturin develop -r` by passing `build` option to
    // `./test.sh`.
    let root = get_project_root().unwrap();
    let target_mtime = mtime_recursive(&root.join("target/")).unwrap();
    // TODO: Better not hard-code venv dir `,env`.
    let build = mtime_recursive(&Path::new(".env")).map_or("build", |src_mtime| {
        if target_mtime > src_mtime {
            "build"
        } else {
            ""
        }
    });
    assert!(std::process::Command::new("bash")
        .arg("./test.sh")
        .arg(build)
        .spawn()
        .unwrap()
        .wait()
        .unwrap()
        .success())
}

#[cfg(test)]
fn get_project_root() -> io::Result<PathBuf> {
    let path = env::current_dir()?;
    let mut path_ancestors = path.as_path().ancestors();

    while let Some(p) = path_ancestors.next() {
        let has_cargo = read_dir(p)?
            .into_iter()
            .any(|p| p.unwrap().file_name() == OsString::from("Cargo.lock"));
        if has_cargo {
            return Ok(PathBuf::from(p));
        }
    }

    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "Cargo.lock not found",
    ))
}
