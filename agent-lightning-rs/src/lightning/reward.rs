/// Reward Shaping — automatic intermediate reward generation.
///
/// One of Agent Lightning's key features: automatically generate
/// intermediate rewards for long-horizon tasks to speed up learning.
/// Also converts agent errors into negative reward signals.
use crate::lightning::pomdp::{Observation, RewardSource};

// ─── Reward Shaper ────────────────────────────────────────────────────────────

pub struct RewardShaper {
    pub potential_scale: f64,      // Scale for potential-based shaping
    pub error_penalty: f64,        // Penalty for each agent error
    pub intermediate_scale: f64,   // Scale for auto-generated intermediate rewards
    pub progress_memory: Vec<f64>, // Track potential of previous states
}

impl RewardShaper {
    pub fn new() -> Self {
        RewardShaper {
            potential_scale: 0.1,
            error_penalty: -1.0,
            intermediate_scale: 0.5,
            progress_memory: Vec::new(),
        }
    }

    /// Shape raw reward with potential-based intermediate rewards
    /// F(s, a, s') = γ * Φ(s') - Φ(s)
    pub fn shape(
        &mut self,
        raw_reward: f64,
        current_obs: &Observation,
        next_obs: &Observation,
        gamma: f64,
        source: &RewardSource,
    ) -> f64 {
        let base = match source {
            RewardSource::ErrorPenalty => raw_reward + self.error_penalty,
            RewardSource::UserFeedback => raw_reward * 2.0, // Amplify human signal
            RewardSource::ToolSuccess => raw_reward + 0.1,  // Small bonus for tool use
            _ => raw_reward,
        };

        // Potential-based shaping: Φ(s) = mean of state features
        let phi_current: f64 =
            current_obs.raw.iter().sum::<f64>() / current_obs.raw.len().max(1) as f64;
        let phi_next: f64 = next_obs.raw.iter().sum::<f64>() / next_obs.raw.len().max(1) as f64;

        let shaping = self.potential_scale * (gamma * phi_next - phi_current);
        base + self.intermediate_scale * shaping
    }

    /// Convert agent error to negative reward signal
    pub fn error_to_reward(&self, recoverable: bool) -> f64 {
        if recoverable {
            self.error_penalty * 0.5
        } else {
            self.error_penalty
        }
    }

    /// Automatic intermediate reward for multi-step tasks:
    /// measures cosine similarity progress toward goal embedding
    pub fn progress_reward(&self, state: &[f64], goal: &[f64]) -> f64 {
        if state.len() != goal.len() || state.is_empty() {
            return 0.0;
        }
        let dot: f64 = state.iter().zip(goal).map(|(a, b)| a * b).sum();
        let norm_s: f64 = state.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_g: f64 = goal.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_s == 0.0 || norm_g == 0.0 {
            return 0.0;
        }
        (dot / (norm_s * norm_g)).clamp(-1.0, 1.0) * self.intermediate_scale
    }

    /// AIR — Automatic Intermediate Rewarding (from paper)
    /// Converts a sequence of system/tool signals into a reward vector.
    pub fn air_from_signals(&self, signals: &[bool]) -> Vec<f64> {
        signals
            .iter()
            .map(|&success| {
                if success {
                    self.intermediate_scale // Tool success bonus
                } else {
                    self.error_penalty // Tool failure penalty
                }
            })
            .collect()
    }
}

impl Default for RewardShaper {
    fn default() -> Self {
        Self::new()
    }
}
