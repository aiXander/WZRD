// Thin entry point — the actual setup lives in `lib.rs` so the same code
// can be invoked from `cargo tauri dev`, the bundled app, or tests.
#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

fn main() {
    wzrd_app_lib::run()
}
