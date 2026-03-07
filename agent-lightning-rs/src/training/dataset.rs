/// Training Dataset — for offline reinforcement learning and replay buffer.
///
/// Stores episodes as collections of Transitions.
/// Supports sampling for experience replay.
use crate::rl::transition::Transition;

pub struct TrainingDataset {
    pub episodes: Vec<Vec<Transition>>,
    pub max_episodes: usize,
}

impl TrainingDataset {
    pub fn new(max_episodes: usize) -> Self {
        Self {
            episodes: Vec::new(),
            max_episodes,
        }
    }

    /// Add a new episode to the dataset (FIFO if full)
    pub fn add_episode(&mut self, episode: Vec<Transition>) {
        if episode.is_empty() {
            return;
        }
        if self.episodes.len() >= self.max_episodes {
            self.episodes.remove(0);
        }
        self.episodes.push(episode);
    }

    /// Sample a random episode from the dataset
    pub fn sample_episode(&self) -> Option<&Vec<Transition>> {
        if self.episodes.is_empty() {
            return None;
        }
        let idx = crate::rand_usize(self.episodes.len());
        Some(&self.episodes[idx])
    }

    /// Sample a batch of episodes
    pub fn sample_batch(&self, batch_size: usize) -> Vec<Vec<Transition>> {
        let mut batch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            if let Some(ep) = self.sample_episode() {
                batch.push(ep.clone());
            }
        }
        batch
    }

    pub fn len(&self) -> usize {
        self.episodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }
}
