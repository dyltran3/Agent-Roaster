/// Agent-Roaster Core Library
pub mod agents;
pub mod guardrails;
pub mod realtime;
pub mod providers;
pub mod memory;

// Existing modules from agent-lightning (retained for now or integration)
pub mod core;
pub mod envs;
pub mod lightning;
pub mod nn;
pub mod rl;
pub mod security;
pub mod training;
pub mod ui;

/// Built-in logging macro (replaces the `log` crate)
#[macro_export]
macro_rules! lightning_log {
    (info, $($arg:tt)*) => { println!("[INFO]  {}", format_args!($($arg)*)) };
    (warn, $($arg:tt)*) => { println!("[WARN]  {}", format_args!($($arg)*)) };
    (error, $($arg:tt)*) => { eprintln!("[ERROR] {}", format_args!($($arg)*)) };
}

/// Built-in PRNG — XorShift64, zero external deps, thread-safe via thread_local.
/// Drop-in replacement for rand::thread_rng() + gen::<f64>()
pub fn rand_f64() -> f64 {
    use std::cell::Cell;
    thread_local! {
        // Seed from current time in nanoseconds for variety between runs
        static STATE: Cell<u64> = Cell::new({
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.subsec_nanos() as u64 ^ (d.as_secs() << 32))
                .unwrap_or(0xdeadbeef_cafebabe)
        });
    }
    STATE.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        // Map to [0.0, 1.0)
        (x >> 11) as f64 / (1u64 << 53) as f64
    })
}

/// Random integer in [0, n)
pub fn rand_usize(n: usize) -> usize {
    (rand_f64() * n as f64) as usize
}

/// Random f64 in [lo, hi)
pub fn rand_range(lo: f64, hi: f64) -> f64 {
    lo + rand_f64() * (hi - lo)
}
