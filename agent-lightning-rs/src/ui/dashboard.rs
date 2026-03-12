/// ASCII Training Dashboard for Agent Lightning.
/// Uses ANSI escape sequences for a zero-dependency, real-time terminal UI.
use std::io::{self, Write};

pub struct Dashboard {
    pub episodes: u64,
    pub max_episodes: u64,
    pub cur_reward: f64,
    pub mean_reward: f64,
    pub loss: f64,
    pub throughput: f64,
    pub policy_version: u32,
    pub status: String,
    history_rewards: Vec<f64>,
}

impl Default for Dashboard {
    fn default() -> Self {
        Self::new(0)
    }
}

impl Dashboard {
    pub fn new(max_episodes: u64) -> Self {
        Dashboard {
            episodes: 0,
            max_episodes,
            cur_reward: 0.0,
            mean_reward: 0.0,
            loss: 0.0,
            throughput: 0.0,
            policy_version: 0,
            status: "Initializing...".to_string(),
            history_rewards: Vec::new(),
        }
    }

    /// Clear the screen and reset cursor.
    pub fn clear(&self) {
        print!("\x1B[2J\x1B[H");
        io::stdout().flush().unwrap();
    }

    /// Update dashboard internal state.
    pub fn update(&mut self, reward: f64, loss: f64, version: u32) {
        self.episodes += 1;
        self.cur_reward = reward;
        self.loss = loss;
        self.policy_version = version;

        self.history_rewards.push(reward);
        if self.history_rewards.len() > 20 {
            self.history_rewards.remove(0);
        }

        // Simple moving average
        let sum: f64 = self.history_rewards.iter().sum();
        self.mean_reward = sum / self.history_rewards.len() as f64;
    }

    /// Render the dashboard to the terminal.
    pub fn render(&self) {
        let mut stdout = io::stdout();

        // ANSI Color Constants
        let c_border = "\x1B[38;5;239m"; // Dark gray border
        let c_title = "\x1B[1;36m"; // Bright cyan
        let c_hl = "\x1B[1;33m"; // Yellow highlight
        let c_text = "\x1B[97m"; // Bright white text
        let c_reset = "\x1B[0m";

        let reward_color = if self.cur_reward > 0.0 {
            "\x1B[1;32m"
        } else {
            "\x1B[1;31m"
        };

        // Top Border with Title
        writeln!(
            stdout,
            "\x1B[H{}╭─────────────────────────────────────────────────────────────╮{}",
            c_border, c_reset
        )
        .unwrap();
        writeln!(
            stdout,
            "{}│{}  ⚡ AGENT LIGHTNING {}Training Dashboard \x1B[32mv2.0         {}│{}",
            c_border, c_title, c_text, c_border, c_reset
        )
        .unwrap();
        writeln!(
            stdout,
            "{}├─────────────────────────────────────────────────────────────┤{}",
            c_border, c_reset
        )
        .unwrap();

        // Section 1: Progress & Status
        let progress = if self.max_episodes > 0 {
            (self.episodes as f64 / self.max_episodes as f64).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let bar_width = 30;
        let filled = (progress * bar_width as f64) as usize;
        let empty = bar_width - filled;
        let p_bar = format!(
            "\x1B[42m{}\x1B[0m\x1B[48;5;238m{}\x1B[0m",
            " ".repeat(filled),
            " ".repeat(empty)
        );

        writeln!(
            stdout,
            "{}│{}  Status:     {} {:0.1}%",
            c_border,
            c_hl,
            p_bar,
            progress * 100.0
        )
        .unwrap();
        writeln!(
            stdout,
            "{}│{}  Mode:       {} {:<37} {}│{}",
            c_border, c_text, c_hl, self.status, c_border, c_reset
        )
        .unwrap();
        writeln!(
            stdout,
            "{}│{}  Episodes:   {} {} / {} ({:.2} ep/s)               {}│{}",
            c_border,
            c_text,
            c_reset,
            self.episodes,
            self.max_episodes,
            self.throughput,
            c_border,
            c_reset
        )
        .unwrap();

        writeln!(
            stdout,
            "{}├─────────────────────────────────────────────────────────────┤{}",
            c_border, c_reset
        )
        .unwrap();

        // Section 2: Metrics
        writeln!(
            stdout,
            "{}│{}  Current Reward: {}{:>10.2}{} | Mean (L20): \x1B[34m{:>10.2}{}",
            c_border, c_text, reward_color, self.cur_reward, c_text, self.mean_reward, c_reset
        )
        .unwrap();
        writeln!(
            stdout,
            "{}│{}  Policy Loss:    \x1B[35m{:>10.4}{} | Policy Ver: \x1B[33mv{:<5}{}",
            c_border, c_text, self.loss, c_text, self.policy_version, c_reset
        )
        .unwrap();

        writeln!(
            stdout,
            "{}├─────────────────────────────────────────────────────────────┤{}",
            c_border, c_reset
        )
        .unwrap();

        // Chart Header
        writeln!(
            stdout,
            "{}│{}  Reward Trend (Last 20)                              {}│{}",
            c_border, c_text, c_border, c_reset
        )
        .unwrap();
        self.render_chart(&mut stdout);

        writeln!(
            stdout,
            "{}╰─────────────────────────────────────────────────────────────╯{}",
            c_border, c_reset
        )
        .unwrap();
        stdout.flush().unwrap();
    }

    fn render_chart(&self, stdout: &mut io::Stdout) {
        if self.history_rewards.is_empty() {
            writeln!(
                stdout,
                "│                                                             │"
            )
            .unwrap();
            return;
        }

        let height = 5;
        let min = self
            .history_rewards
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max = self
            .history_rewards
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let diff = if (max - min).abs() < 1e-6 {
            1.0
        } else {
            max - min
        };

        for y in (0..height).rev() {
            write!(stdout, "\x1B[38;5;239m│\x1B[0m    ").unwrap();
            let threshold = min + (y as f64 / height as f64) * diff;

            for &val in &self.history_rewards {
                if val >= threshold {
                    write!(stdout, "\x1B[1;32m█\x1B[0m").unwrap();
                } else {
                    write!(stdout, " ").unwrap();
                }
            }
            writeln!(stdout).unwrap();
        }
    }
}
