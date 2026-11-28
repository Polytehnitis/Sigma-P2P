//! Builds the vendored whisper.cpp (whisper, parakeet and ggml's CPU backend) with CMake, and the
//! small C++ interface in `csrc/` that the Rust side calls.
//!
//! Always an optimized build, for any CPU of the target's family: a debug build of whisper.cpp
//! is unusably slow, and a build for the compiling machine's CPU can crash on another one.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let target = env::var("TARGET").unwrap();
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let msvc = target.contains("msvc");
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let whisper = root.join("whisper.cpp");
    println!("cargo:rerun-if-changed=csrc");
    println!("cargo:rerun-if-changed=whisper.cpp");

    let mut cmake = cmake::Config::new(root.join("csrc"));
    cmake
        .profile("Release")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("WHISPER_BUILD_TESTS", "OFF")
        .define("WHISPER_BUILD_EXAMPLES", "OFF")
        .define("WHISPER_BUILD_SERVER", "OFF")
        .define("WHISPER_ALL_WARNINGS", "OFF")
        .define("WHISPER_CURL", "OFF")
        .define("WHISPER_SDL2", "OFF")
        .define("GGML_BUILD_TESTS", "OFF")
        .define("GGML_BUILD_EXAMPLES", "OFF")
        .define("GGML_CCACHE", "OFF")
        .define("GGML_OPENMP", "OFF")
        .define("GGML_METAL", "OFF")
        .define("GGML_NATIVE", "OFF")
        .pic(true);
    if arch == "x86_64" {
        // Every x86-64 CPU of the last decade.
        for feature in ["GGML_AVX", "GGML_AVX2", "GGML_FMA", "GGML_F16C"] {
            cmake.define(feature, "ON");
        }
    }
    if msvc {
        // The cmake crate replaces the Visual Studio generator's Release flags with the C
        // compiler flags of this build script, which have no optimization (and it drops any
        // /O flag given to it). Defined here, it leaves them alone.
        cmake
            .define("CMAKE_C_FLAGS_RELEASE", "/MD /O2 /Ob2 /DNDEBUG")
            .define("CMAKE_CXX_FLAGS_RELEASE", "/MD /O2 /Ob2 /DNDEBUG /utf-8");
    }
    let out = cmake.build();

    // Linked first: it uses whisper and parakeet, which use ggml.
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .opt_level(2)
        .file(root.join("csrc/sigma_asr.cpp"))
        .include(whisper.join("include"))
        .include(whisper.join("ggml/include"))
        .flag_if_supported("/utf-8")
        .compile("sigma_asr");

    let mut ggml_blas = false;
    for dir in ["lib", "lib64"] {
        let lib = out.join(dir);
        if lib.is_dir() {
            println!("cargo:rustc-link-search=native={}", lib.display());
            ggml_blas |= has_lib(&lib, "ggml-blas");
        }
    }
    for lib in ["parakeet", "whisper"] {
        println!("cargo:rustc-link-lib=static={lib}");
    }
    if ggml_blas {
        println!("cargo:rustc-link-lib=static=ggml-blas");
    }
    for lib in ["ggml", "ggml-cpu", "ggml-base"] {
        println!("cargo:rustc-link-lib=static={lib}");
    }

    match os.as_str() {
        "macos" | "ios" => {
            println!("cargo:rustc-link-lib=dylib=c++");
            println!("cargo:rustc-link-lib=framework=Accelerate");
            println!("cargo:rustc-link-lib=framework=Foundation");
        }
        "android" => println!("cargo:rustc-link-lib=dylib=c++_shared"),
        "windows" if msvc => println!("cargo:rustc-link-lib=advapi32"),
        "windows" => println!("cargo:rustc-link-lib=dylib=stdc++"),
        _ => println!("cargo:rustc-link-lib=dylib=stdc++"),
    }
}

/// Whether `dir` holds a static library named `name` (libname.a or name.lib).
fn has_lib(dir: &Path, name: &str) -> bool {
    [format!("lib{name}.a"), format!("{name}.lib")].iter().any(|f| PathBuf::from(dir).join(f).exists())
}
