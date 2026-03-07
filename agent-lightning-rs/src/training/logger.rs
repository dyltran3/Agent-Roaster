//! Training logger — formatted console output for training metrics.

use std::time::Instant;

pub struct Logger {
    pub start_time: Instant,
    pub best_reward: f64,
    pub episode_rewards: Vec<f64>,
}

impl Logger {
    pub fn new() -> Self {
        Logger {
            start_time: Instant::now(),
            best_reward: f64::NEG_INFINITY,
            episode_rewards: Vec::new(),
        }
    }

    pub fn log_episode(&mut self, episode: usize, reward: f64, actor_loss: f64, critic_loss: f64) {
        self.episode_rewards.push(reward);
        if reward > self.best_reward {
            self.best_reward = reward;
        }

        let elapsed = self.start_time.elapsed().as_secs_f64();
        let mean = if self.episode_rewards.len() >= 10 {
            let n = self.episode_rewards.len();
            self.episode_rewards[n - 10..].iter().sum::<f64>() / 10.0
        } else {
            self.episode_rewards.iter().sum::<f64>() / self.episode_rewards.len() as f64
        };

        println!(
            "[Ep {:>4}] reward={:>8.2}  mean10={:>8.2}  best={:>8.2}  actor_loss={:.4}  critic_loss={:.4}  t={:.1}s",
            episode, reward, mean, self.best_reward, actor_loss, critic_loss, elapsed
        );
    }

    pub fn log_eval(&self, episode: usize, eval_reward: f64) {
        println!(
            "\n  ┌─ EVAL [Ep {}] avg_reward={:.2} ─┐",
            episode, eval_reward
        );
    }

    pub fn print_header(&self, algorithm: &str, env_name: &str) {
        println!("\n╔══════════════════════════════════════════════════════════════════╗");
        println!("║   ⚡ Agent Lightning RS — Reinforcement Learning Framework       ║");
        println!(
            "║   Algorithm : {:<20}  Environment : {:<13} ║",
            algorithm, env_name
        );
        println!("╚══════════════════════════════════════════════════════════════════╝\n");
    }

    pub fn print_footer(&self) {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        println!("\n╔════════════════════════════════╗");
        println!("║  Training Complete!             ║");
        println!("║  Best Reward:  {:>10.2}       ║", self.best_reward);
        println!("║  Total Time:   {:>8.1}s        ║", elapsed);
        println!("╚════════════════════════════════╝\n");
    }
}

impl Default for Logger {
    fn default() -> Self {
        Self::new()
    }
}
