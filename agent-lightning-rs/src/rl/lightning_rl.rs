use crate::rl::credit_assignment;
use crate::rl::grpo::GRPOAgent;
use crate::rl::ppo::PPOAgent;
/// LightningRL Orchestrator
///
/// This module implements the high-level logic of the Agent Lightning reinforcement learning algorithm.
/// It combines Credit Assignment (Step 1) and Token-level Policy Optimization (Step 2).
use crate::rl::transition::Transition;

pub enum TrainerType {
    PPO(PPOAgent),
    GRPO(GRPOAgent),
}

pub struct LightningRL {
    pub trainer: TrainerType,
    pub config: LightningRLConfig,
}

pub struct LightningRLConfig {
    pub credit_mode: CreditMode,
    pub gamma: f64,
}

#[derive(Debug, Clone, Copy)]
pub enum CreditMode {
    Uniform,
    Discounted,
    GAE,
}

impl LightningRL {
    pub fn new(trainer: TrainerType, config: LightningRLConfig) -> Self {
        Self { trainer, config }
    }

    /// The core training algorithm step from the paper:
    /// 1. Assign credit (R -> Advantages)
    /// 2. Update parameters (Token-level Optim)
    pub fn train_on_episodes(&mut self, mut episodes: Vec<Vec<Transition>>) -> f64 {
        if episodes.is_empty() {
            return 0.0;
        }

        let mut all_transitions = Vec::new();

        // Step 1: Credit Assignment for each episode
        for episode in episodes.iter_mut() {
            match self.config.credit_mode {
                CreditMode::Uniform => credit_assignment::assign_uniform_credit(episode),
                CreditMode::Discounted => {
                    credit_assignment::assign_discounted_credit(episode, self.config.gamma)
                }
                CreditMode::GAE => {
                    // For GAE, we need value estimates.
                    // This is currently handled inside the specific PPO implementation if needed,
                    // If called generally, fallback to Discounted to avoid Uniform explosions.
                    credit_assignment::assign_discounted_credit(episode, self.config.gamma);
                }
            }
            all_transitions.extend(episode.clone());
        }

        // Normalize advantages across the whole batch
        credit_assignment::normalize_advantages(&mut all_transitions);

        // Step 2: Policy Update (Token-level)
        match &mut self.trainer {
            TrainerType::PPO(agent) => {
                let (actor_loss, _critic_loss) = agent.update_from_transitions(&all_transitions);
                actor_loss
            }
            TrainerType::GRPO(agent) => agent.update_from_transitions(&all_transitions),
        }
    }
}
