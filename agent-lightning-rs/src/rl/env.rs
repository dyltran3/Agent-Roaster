//! Environment trait and types — POMDP abstraction for any RL environment.
//!
//! Inspired by Microsoft Agent Lightning's POMDP abstraction,
//! allowing any environment to plug into the training framework.

use crate::core::tensor::Tensor;

// ─── Core Types ───────────────────────────────────────────────────────────────

pub type State = Vec<f64>;
pub type Action = usize;

#[derive(Debug, Clone)]
pub struct StepResult {
    pub next_state: State,
    pub reward: f64,
    pub done: bool,
    pub info: String,
}

// ─── Environment Trait ────────────────────────────────────────────────────────

/// Core environment interface (Gym-like)
pub trait Environment: Send {
    /// Reset environment to initial state, return first observation
    fn reset(&mut self) -> State;

    /// Take action, return (next_state, reward, done, info)
    fn step(&mut self, action: Action) -> StepResult;

    /// Number of discrete actions available
    fn action_space(&self) -> usize;

    /// Dimension of the state/observation vector
    fn state_space(&self) -> usize;

    /// Current state (without advancing)
    fn current_state(&self) -> State;

    /// Human-readable name
    fn name(&self) -> &str;

    /// Optional: render current state to console
    fn render(&self) {}
}

// ─── POMDP State (Lightning abstraction) ─────────────────────────────────────

/// Represents a single step in a Partially Observable Markov Decision Process.
/// This is the fundamental unit that Lightning Client sends to Lightning Server.
#[derive(Debug, Clone)]
pub struct POMDPStep {
    pub episode_id: u64,
    pub step_id: u64,
    pub observation: State,
    pub action: Action,
    pub reward: f64,
    pub next_observation: State,
    pub done: bool,
    pub log_prob: f64,       // Log probability of the selected action
    pub value_estimate: f64, // Value function estimate for this state
    pub info: String,
}

impl POMDPStep {
    pub fn to_tensor_obs(&self) -> Tensor {
        Tensor::new(self.observation.clone(), vec![self.observation.len()])
    }

    pub fn to_tensor_next_obs(&self) -> Tensor {
        Tensor::new(
            self.next_observation.clone(),
            vec![self.next_observation.len()],
        )
    }
}

// ─── Trajectory (Episode rollout) ────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Trajectory {
    pub episode_id: u64,
    pub steps: Vec<POMDPStep>,
    pub total_reward: f64,
}

impl Trajectory {
    pub fn new(episode_id: u64) -> Self {
        Trajectory {
            episode_id,
            steps: Vec::new(),
            total_reward: 0.0,
        }
    }

    pub fn push(&mut self, step: POMDPStep) {
        self.total_reward += step.reward;
        self.steps.push(step);
    }

    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Compute discounted returns (for advantage estimation)
    pub fn compute_returns(&self, gamma: f64) -> Vec<f64> {
        let n = self.steps.len();
        let mut returns = vec![0.0f64; n];
        let mut running = 0.0;
        for i in (0..n).rev() {
            running = self.steps[i].reward + gamma * running;
            returns[i] = running;
        }
        returns
    }

    /// GAE (Generalized Advantage Estimation)
    pub fn compute_advantages(&self, gamma: f64, lambda: f64) -> Vec<f64> {
        let n = self.steps.len();
        let mut advantages = vec![0.0f64; n];
        let mut gae = 0.0;
        for i in (0..n).rev() {
            let next_value = if i + 1 < n {
                self.steps[i + 1].value_estimate
            } else {
                0.0
            };
            let delta = self.steps[i].reward + gamma * next_value - self.steps[i].value_estimate;
            gae = delta + gamma * lambda * gae;
            advantages[i] = gae;
        }
        advantages
    }
}
