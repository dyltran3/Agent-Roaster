/// ASCII Training Dashboard for Agent Lightning.
/// Uses ANSI escape sequences for a zero-dependency, real-time terminal UI.
use std::io::{self, Write};

pub struct Dashboard {
    pub episodes: u64,
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
        Self::new()
    }
}

impl Dashboard {
    pub fn new() -> Self {
        Dashboard {
            episodes: 0,
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

        // Header
        writeln!(stdout, "\x1B[H\x1B[1;36m═══════════════════════════════════════════════════════════════\x1B[0m").unwrap();
        writeln!(
            stdout,
            "\x1B[1;37m  AGENT LIGHTNING \x1B[1;33m[Training Dashboard]\x1B[0m"
        )
        .unwrap();
        writeln!(
            stdout,
            "\x1B[1;36m═══════════════════════════════════════════════════════════════\x1B[0m"
        )
        .unwrap();

        // Metrics Table
        writeln!(
            stdout,
            "  \x1B[1;37mStatus:\x1B[0m {:<15} | \x1B[1;37mVersion:\x1B[0m v{:<5}",
            self.status, self.policy_version
        )
        .unwrap();
        writeln!(
            stdout,
            "  \x1B[1;37mEpisodes:\x1B[0m {:<13} | \x1B[1;37mThroughput:\x1B[0m {:.2} ep/s",
            self.episodes, self.throughput
        )
        .unwrap();
        writeln!(
            stdout,
            "  \x1B[1;36m───────────────────────────────────────────────────────────────\x1B[0m"
        )
        .unwrap();

        // Values
        let reward_color = if self.cur_reward > 0.0 {
            "\x1B[1;32m"
        } else {
            "\x1B[1;31m"
        };
        writeln!(
            stdout,
            "  Current Reward: {}{:>10.2}\x1B[0m | Mean (last 20): \x1B[1;34m{:>10.2}\x1B[0m",
            reward_color, self.cur_reward, self.mean_reward
        )
        .unwrap();
        writeln!(
            stdout,
            "  Policy Loss:    \x1B[1;35m{:>10.4}\x1B[0m",
            self.loss
        )
        .unwrap();
        writeln!(
            stdout,
            "  \x1B[1;36m───────────────────────────────────────────────────────────────\x1B[0m"
        )
        .unwrap();

        // ASCII Chart for Rewards
        writeln!(stdout, "  \x1B[1;37mReward Trend (last 20):\x1B[0m").unwrap();
        self.render_chart(&mut stdout);

        writeln!(
            stdout,
            "\x1B[1;36m═══════════════════════════════════════════════════════════════\x1B[0m"
        )
        .unwrap();
        stdout.flush().unwrap();
    }

    fn render_chart(&self, stdout: &mut io::Stdout) {
        if self.history_rewards.is_empty() {
            return;
        }

        let height = 5;
        let _width = 40;
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
            write!(stdout, "    ").unwrap();
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
