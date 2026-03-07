use crate::lightning::client::{ClientToServerMsg, ServerToClientMsg};
use crate::lightning_log;
use crate::rl::lightning_rl::LightningRL;
use crate::rl::transition::Transition;
/// Lightning Server — orchestrates training from accumulated agent experience.
///
/// Uses std::sync::mpsc (synchronous) — no tokio/async required.
use std::collections::VecDeque;
use std::sync::mpsc::{Receiver, Sender};

#[derive(Debug, Default, Clone)]
pub struct ServerStats {
    pub total_steps: u64,
    pub total_episodes: u64,
    pub updates_performed: u64,
    pub last_episode_reward: f64,
    pub mean_reward_100: f64,
    pub policy_version: u64,
}

pub struct LightningServer {
    pub stats: ServerStats,
    pub trainer: Option<LightningRL>,
    pub policy_version: u32,
    episode_buffer: Vec<Vec<Transition>>,
    current_episode: Vec<Transition>,
    error_log: Vec<(u64, u64, String)>,
    reward_history: VecDeque<f64>,
    reward_history_cap: usize,
    pub update_interval_episodes: usize,
    client_senders: Vec<Sender<ServerToClientMsg>>,
}

impl LightningServer {
    pub fn new(update_interval_episodes: usize) -> Self {
        LightningServer {
            stats: ServerStats::default(),
            trainer: None,
            policy_version: 0,
            episode_buffer: Vec::new(),
            current_episode: Vec::new(),
            error_log: Vec::new(),
            reward_history: VecDeque::new(),
            reward_history_cap: 100,
            update_interval_episodes,
            client_senders: Vec::new(),
        }
    }

    pub fn register_client(&mut self, sender: Sender<ServerToClientMsg>) {
        self.client_senders.push(sender);
    }

    /// Process all pending messages from clients (synchronous, non-blocking drain)
    /// Process all pending messages from clients
    pub fn process_messages(&mut self, receiver: &Receiver<ClientToServerMsg>) {
        while let Ok(msg) = receiver.try_recv() {
            self.handle_message(msg);
        }

        if self.episode_buffer.len() >= self.update_interval_episodes {
            self.trigger_training_update();
        }
    }

    pub fn handle_message(&mut self, msg: ClientToServerMsg) {
        match msg {
            ClientToServerMsg::Transition(t) => {
                self.stats.total_steps += t.rewards.len() as u64;
                self.current_episode.push(t);
            }
            ClientToServerMsg::Error {
                episode_id,
                step_id,
                description,
            } => {
                lightning_log!(
                    warn,
                    "[Server] Error ep={} step={}: {}",
                    episode_id,
                    step_id,
                    description
                );
                self.error_log.push((episode_id, step_id, description));
            }
            ClientToServerMsg::EpisodeDone {
                episode_id,
                total_reward,
                intermediate_rewards: _,
            } => {
                self.stats.total_episodes += 1;
                self.stats.last_episode_reward = total_reward;

                // Finalize the episode if we have accumulated transitions
                if !self.current_episode.is_empty() {
                    // AIR strategy: if intermediate rewards are passed, merge them into transitions
                    // Typically the client might pass some signals, or we use a RewardShaper.
                    // For now, we take the provided rewards.

                    self.episode_buffer
                        .push(std::mem::take(&mut self.current_episode));
                }

                self.reward_history.push_back(total_reward);
                if self.reward_history.len() > self.reward_history_cap {
                    self.reward_history.pop_front();
                }
                self.stats.mean_reward_100 =
                    self.reward_history.iter().sum::<f64>() / self.reward_history.len() as f64;
                lightning_log!(
                    info,
                    "[Server] Ep {} done  reward={:.2}  mean100={:.2}",
                    episode_id,
                    total_reward,
                    self.stats.mean_reward_100
                );
            }
        }
    }

    pub fn trigger_training_update(&mut self) {
        if self.trainer.is_none() || self.episode_buffer.is_empty() {
            return;
        }

        let episodes = std::mem::take(&mut self.episode_buffer);
        let _n_episodes = episodes.len();

        // Step 1 & 2 facilitated by LightningRL orchestrator
        let avg_loss = if let Some(trainer) = &mut self.trainer {
            trainer.train_on_episodes(episodes)
        } else {
            0.0
        };

        self.stats.updates_performed += 1;
        self.stats.policy_version += 1;
        let version = self.stats.policy_version;

        lightning_log!(
            info,
            "[Server] Update #{} avg_loss={:.4} policy_v={}",
            self.stats.updates_performed,
            avg_loss,
            version
        );

        for sender in &self.client_senders {
            let _ = sender.send(ServerToClientMsg::PolicyUpdate {
                params: vec![avg_loss],
                version,
            });
        }
    }

    pub fn print_stats(&self) {
        println!("\n╔══════════════════════════════════════════╗");
        println!("║       ⚡ Lightning Server Stats           ║");
        println!("╠══════════════════════════════════════════╣");
        println!(
            "║  Total Steps:     {:>8}                ║",
            self.stats.total_steps
        );
        println!(
            "║  Total Episodes:  {:>8}                ║",
            self.stats.total_episodes
        );
        println!(
            "║  Updates Done:    {:>8}                ║",
            self.stats.updates_performed
        );
        println!(
            "║  Policy Version:  {:>8}                ║",
            self.stats.policy_version
        );
        println!(
            "║  Mean Reward/100: {:>8.2}                ║",
            self.stats.mean_reward_100
        );
        println!("╚══════════════════════════════════════════╝\n");
    }
}
