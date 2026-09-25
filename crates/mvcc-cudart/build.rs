//! Compiles the Objective-C++ Metal layer (cpp/mvcc-metal) into the runtime and sets the dylib install name.

fn main() {
    println!("cargo:rerun-if-changed=../../cpp/mvcc-metal/mvcc_metal.mm");
    println!("cargo:rerun-if-changed=../../cpp/mvcc-metal/mvcc_metal.h");
    println!("cargo:rerun-if-changed=../../cpp/mvcc-metal/mvcc_props.cpp");
    println!("cargo:rerun-if-changed=../../cpp/mvcc-metal/mvcc_autolink.s");
    println!("cargo:rerun-if-changed=../../include/driver_types.h");
    cc::Build::new()
        .file("../../cpp/mvcc-metal/mvcc_metal.mm")
        .file("../../cpp/mvcc-metal/mvcc_props.cpp")
        .file("../../cpp/mvcc-metal/mvcc_autolink.s")
        .include("../../include")
        .flag("-fobjc-arc")
        .flag("-mmacosx-version-min=26.0")
        .flag("-std=c++17")
        .flag("-Wno-unused-parameter")
        .cpp(true)
        .compile("mvcc_metal");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=IOKit");
    println!("cargo:rustc-link-lib=c++");
    // install name so linked programs find us through rpath. headerpad lets
    // install_toolkit.sh rewrite the id to the (long) gem toolkit path.
    println!("cargo:rustc-cdylib-link-arg=-Wl,-install_name,@rpath/libcudart.dylib");
    println!("cargo:rustc-cdylib-link-arg=-Wl,-headerpad_max_install_names");
}
