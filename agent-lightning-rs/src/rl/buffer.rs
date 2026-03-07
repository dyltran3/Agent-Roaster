/// Experience Replay Buffer — stores and samples agent transitions.
///
/// Two variants:
///   - ReplayBuffer: standard circular buffer for DQN-style off-policy
///   - RolloutBuffer: collects entire on-policy rollouts for PPO/GRPO
use crate::rl::env::POMDPStep;
use crate::rl::transition::Transition;

// ─── Standard Replay Buffer (off-policy) ─────────────────────────────────────

pub struct ReplayBuffer {
    pub capacity: usize,
    pub buffer: Vec<POMDPStep>,
    pub position: usize,
    pub full: bool,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        ReplayBuffer {
            capacity,
            buffer: Vec::with_capacity(capacity),
            position: 0,
            full: false,
        }
    }

    pub fn push(&mut self, step: POMDPStep) {
        if self.buffer.len() < self.capacity {
            self.buffer.push(step);
        } else {
            self.buffer[self.position] = step;
            self.full = true;
        }
        self.position = (self.position + 1) % self.capacity;
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn can_sample(&self, batch_size: usize) -> bool {
        self.len() >= batch_size
    }

    /// Sample a random mini-batch of transitions
    pub fn sample(&self, batch_size: usize) -> Vec<&POMDPStep> {
        assert!(self.can_sample(batch_size), "Not enough samples in buffer");
        let mut indices: Vec<usize> = (0..self.len()).collect();
        // Fisher-Yates shuffle
        for i in (1..indices.len()).rev() {
            let j = crate::rand_usize(i + 1);
            indices.swap(i, j);
        }
        indices[..batch_size]
            .iter()
            .map(|&i| &self.buffer[i])
            .collect()
    }
}

// ─── Rollout Buffer (on-policy: PPO/GRPO) ────────────────────────────────────

/// Temporarily stores a single on-policy rollout.
/// Cleared after each update epoch.
pub struct RolloutBuffer {
    pub steps: Vec<POMDPStep>,
    pub returns: Vec<f64>,
    pub advantages: Vec<f64>,
}

impl RolloutBuffer {
    pub fn new() -> Self {
        RolloutBuffer {
            steps: Vec::new(),
            returns: Vec::new(),
            advantages: Vec::new(),
        }
    }

    pub fn push(&mut self, step: POMDPStep) {
        self.steps.push(step);
    }

    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Compute and store returns + advantages from stored steps
    pub fn finalize(&mut self, gamma: f64, lambda: f64) {
        let n = self.steps.len();
        self.returns = vec![0.0; n];
        self.advantages = vec![0.0; n];

        let mut running_return = 0.0;
        let mut gae = 0.0;

        for i in (0..n).rev() {
            let next_value = if i + 1 < n {
                self.steps[i + 1].value_estimate
            } else {
                0.0
            };
            let done_mask = if self.steps[i].done { 0.0 } else { 1.0 };

            running_return = self.steps[i].reward + gamma * done_mask * running_return;
            self.returns[i] = running_return;

            let delta = self.steps[i].reward + gamma * done_mask * next_value
                - self.steps[i].value_estimate;
            gae = delta + gamma * lambda * done_mask * gae;
            self.advantages[i] = gae;
        }

        // Normalize advantages for training stability
        let adv_mean = self.advantages.iter().sum::<f64>() / n as f64;
        let adv_var = self
            .advantages
            .iter()
            .map(|x| (x - adv_mean).powi(2))
            .sum::<f64>()
            / n as f64;
        let adv_std = (adv_var + 1e-8).sqrt();
        for a in self.advantages.iter_mut() {
            *a = (*a - adv_mean) / adv_std;
        }
    }

    /// Sample random mini-batch of (step_idx) for PPO update epochs
    pub fn sample_indices(&self, batch_size: usize) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.steps.len()).collect();
        for i in (1..indices.len()).rev() {
            let j = crate::rand_usize(i + 1);
            indices.swap(i, j);
        }
        let take = batch_size.min(indices.len());
        indices[..take].to_vec()
    }

    pub fn clear(&mut self) {
        self.steps.clear();
        self.returns.clear();
        self.advantages.clear();
    }
}

impl Default for RolloutBuffer {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Episode Buffer (Agent Lightning specific) ───────────────────────────────

/// Stores complete episodes as sequences of Unified MDP Transitions.
pub struct EpisodeBuffer {
    pub episodes: Vec<Vec<Transition>>,
}

impl EpisodeBuffer {
    pub fn new() -> Self {
        Self {
            episodes: Vec::new(),
        }
    }

    pub fn push_episode(&mut self, episode: Vec<Transition>) {
        self.episodes.push(episode);
    }

    pub fn clear(&mut self) {
        self.episodes.clear();
    }

    pub fn len(&self) -> usize {
        self.episodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }

    /// Flatten all transitions from all episodes into a single batch for training
    pub fn all_transitions(&self) -> Vec<Transition> {
        self.episodes.iter().flatten().cloned().collect()
    }
}

impl Default for EpisodeBuffer {
    fn default() -> Self {
        Self::new()
    }
}
