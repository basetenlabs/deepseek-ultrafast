use std::{env, fs, path::PathBuf, process::Command};

fn main() {
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let nvcc = env::var("NVCC").unwrap_or_else(|_| "/usr/local/cuda/bin/nvcc".into());
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=DSUF_SM");
    let arch = format!(
        "-arch={}",
        env::var("DSUF_SM").unwrap_or_else(|_| "sm_100a".into())
    );
    println!("cargo:rerun-if-changed=cu");
    let mut entries = Vec::new();
    let mut srcs: Vec<_> = fs::read_dir("cu")
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    let variant = |p: &PathBuf| p.file_stem().and_then(|s| s.to_str()).is_some_and(|s| s.starts_with("variant_"));
    srcs.sort_by(|a, b| (variant(a), a).cmp(&(variant(b), b)));
    let jobs = env::var("NUM_JOBS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
        .unwrap_or(1)
        .max(1);
    let mut running: std::collections::VecDeque<(PathBuf, std::process::Child)> = Default::default();
    let wait = |(src, mut child): (PathBuf, std::process::Child)| {
        let st = child.wait().expect("nvcc");
        assert!(st.success(), "nvcc failed on {}", src.display());
    };
    for src in srcs {
        println!("cargo:rerun-if-changed={}", src.display());
        if src.extension().and_then(|e| e.to_str()) != Some("cu") || (variant(&src) && arch != "-arch=sm_103a") {
            continue;
        }
        let stem = src.file_stem().unwrap().to_str().unwrap();
        let cubin = out.join(format!("{stem}.cubin"));
        if running.len() >= jobs {
            wait(running.pop_front().unwrap());
        }
        let child = Command::new(&nvcc)
            .args([&arch, "-cubin", "-O3", "-std=c++17", "-lineinfo", "-Icu"])
            .args(["-diag-suppress", "177"])
            .arg(&src)
            .arg("-o")
            .arg(&cubin)
            .spawn()
            .expect("nvcc");
        running.push_back((src.clone(), child));
        entries.push(format!(
            "({stem:?}, include_bytes!({:?}) as &[u8])",
            cubin.to_str().unwrap()
        ));
    }
    running.into_iter().for_each(wait);
    fs::write(out.join("cubins.rs"), format!("pub const CUBINS: &[(&str, &[u8])] = &[{}];\n", entries.join(",\n"))).unwrap();
}
