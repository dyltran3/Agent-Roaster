/// Unified MDP Transition structure based on Agent Lightning paper.
///
/// Instead of per-step transitions, we model agent reasoning as transitions
/// between (Input Prompt, Output Sequence) with per-step internal rewards.
use crate::core::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct Transition {
    /// The input state or prompt context (S_t)
    pub input: Tensor,

    /// The output action or reasoning sequence (A_t)
    /// Usually a sequence of tokens/logits in LLM context
    pub output: Tensor,

    /// Per-token or per-step log-probabilities from the sampling policy
    /// Used for importance sampling in PPO/GRPO.
    pub log_probs: Vec<f64>,

    /// Per-step local rewards (including AIR - Automatic Intermediate Rewards)
    /// and the final terminal reward.
    pub rewards: Vec<f64>,

    /// Total advantage calculated during credit assignment (Step 1 of LightningRL)
    pub advantages: Vec<f64>,
}

impl Transition {
    pub fn new(input: Tensor, output: Tensor, log_probs: Vec<f64>, rewards: Vec<f64>) -> Self {
        let n = rewards.len();
        Self {
            input,
            output,
            log_probs,
            rewards,
            advantages: vec![0.0; n],
        }
    }

    /// Total scalar reward for this transition (terminal + sum of intermediate)
    pub fn total_reward(&self) -> f64 {
        self.rewards.iter().sum()
    }
}

/// A batch of transitions used for a single training update step.
#[derive(Debug, Clone)]
pub struct TransitionBatch {
    pub transitions: Vec<Transition>,
}

impl TransitionBatch {
    pub fn new(transitions: Vec<Transition>) -> Self {
        Self { transitions }
    }

    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }

    pub fn len(&self) -> usize {
        self.transitions.len()
    }
}
