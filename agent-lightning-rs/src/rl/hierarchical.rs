//! Hierarchical RL (LightningRL) — the core algorithm of Microsoft Agent Lightning.
//!
//! Decomposes complex multi-step tasks into sub-goals:
//!   - High-level policy: selects sub-goal / sub-task
//!   - Low-level policy: executes primitive actions to achieve sub-goal
//!   - Credit assignment: reward signal propagated through hierarchy
//!
//! This matches Agent Lightning's approach of subdividing multi-turn,
//! multi-agent interactions into smaller, RL-friendly units.

use crate::core::activation::Activation;
use crate::core::optimizer::{Adam, Optimizer};
use crate::core::tensor::Tensor;
use crate::nn::network::Sequential;
use crate::rl::env::Environment;
use crate::rl::ppo::sample_from_probs;

// ─── Sub-goal definition ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SubGoal {
    pub id: usize,
    pub name: String,
    pub max_steps: usize,
}

// ─── Hierarchical Experience ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct HierarchicalStep {
    pub state: Vec<f64>,
    pub subgoal: usize,
    pub action: usize,
    pub reward: f64,
    pub intrinsic_reward: f64, // Reward for achieving sub-goal
    pub extrinsic_reward: f64, // External environment reward
    pub done: bool,
}

// ─── High-level Policy (Manager) ─────────────────────────────────────────────

pub struct Manager {
    pub policy: Sequential,
    pub optim: Adam,
    pub subgoal_horizon: usize, // Steps before selecting new sub-goal
}

impl Manager {
    pub fn new(state_dim: usize, n_subgoals: usize, lr: f64, horizon: usize) -> Self {
        let policy = Sequential::new()
            .dense(state_dim, 64, Activation::Tanh)
            .dense(64, n_subgoals, Activation::Softmax);

        Manager {
            policy,
            optim: Adam::new(lr),
            subgoal_horizon: horizon,
        }
    }

    pub fn select_subgoal(&self, state: &[f64]) -> usize {
        let t = Tensor::new(state.to_vec(), vec![1, state.len()]);
        let probs = self.policy.forward(&t);
        sample_from_probs(&probs.data)
    }
}

// ─── Low-level Policy (Worker) ────────────────────────────────────────────────

pub struct Worker {
    pub policy: Sequential,
    pub optim: Adam,
}

impl Worker {
    pub fn new(state_dim: usize, n_subgoals: usize, action_dim: usize, lr: f64) -> Self {
        // Input = state concat subgoal one-hot
        let policy = Sequential::new()
            .dense(state_dim + n_subgoals, 64, Activation::Tanh)
            .dense(64, 64, Activation::Tanh)
            .dense(64, action_dim, Activation::Softmax);

        Worker {
            policy,
            optim: Adam::new(lr),
        }
    }

    pub fn select_action(&self, state: &[f64], subgoal: usize, n_subgoals: usize) -> (usize, f64) {
        let mut input = state.to_vec();
        // One-hot encode subgoal
        let mut sg_onehot = vec![0.0f64; n_subgoals];
        sg_onehot[subgoal] = 1.0;
        input.extend_from_slice(&sg_onehot);

        let t = Tensor::new(input, vec![1, state.len() + n_subgoals]);
        let probs = self.policy.forward(&t);
        let action = sample_from_probs(&probs.data);
        let log_prob = (probs.data[action] + 1e-10).ln();
        (action, log_prob)
    }
}

// ─── LightningRL Config ────────────────────────────────────────────────────────

pub struct LightningRLConfig {
    pub lr_manager: f64,
    pub lr_worker: f64,
    pub gamma_manager: f64, // Discount for high-level rewards
    pub gamma_worker: f64,  // Discount for low-level rewards
    pub n_subgoals: usize,
    pub subgoal_horizon: usize,
    pub intrinsic_reward_scale: f64,
    pub max_steps_per_episode: usize,
}

impl Default for LightningRLConfig {
    fn default() -> Self {
        LightningRLConfig {
            lr_manager: 1e-4,
            lr_worker: 3e-4,
            gamma_manager: 0.99,
            gamma_worker: 0.95,
            n_subgoals: 4,
            subgoal_horizon: 10,
            intrinsic_reward_scale: 1.0,
            max_steps_per_episode: 500,
        }
    }
}

// ─── Hierarchical Agent (LightningRL) ─────────────────────────────────────────

pub struct LightningRLAgent {
    pub manager: Manager,
    pub worker: Worker,
    pub config: LightningRLConfig,
    pub history: Vec<HierarchicalStep>,
    pub episode: u64,
}

impl LightningRLAgent {
    pub fn new(state_dim: usize, action_dim: usize, config: LightningRLConfig) -> Self {
        let n_sg = config.n_subgoals;
        let manager = Manager::new(state_dim, n_sg, config.lr_manager, config.subgoal_horizon);
        let worker = Worker::new(state_dim, n_sg, action_dim, config.lr_worker);

        LightningRLAgent {
            manager,
            worker,
            config,
            history: Vec::new(),
            episode: 0,
        }
    }

    /// Run one full episode with hierarchical control
    pub fn run_episode(&mut self, env: &mut dyn Environment) -> f64 {
        let mut state = env.reset();
        let mut total_reward = 0.0;
        let mut step = 0;
        let mut subgoal = self.manager.select_subgoal(&state);
        let mut subgoal_steps = 0;
        let n_sg = self.config.n_subgoals;

        // Manager trajectory (for high-level updates)
        let mut manager_states: Vec<Vec<f64>> = Vec::new();
        let mut manager_subgoals: Vec<usize> = Vec::new();
        let mut manager_rewards: Vec<f64> = Vec::new();

        // Worker trajectory (for low-level updates)
        let mut worker_states: Vec<Vec<f64>> = Vec::new();
        let mut worker_subgoals: Vec<usize> = Vec::new();
        let mut worker_actions: Vec<usize> = Vec::new();
        let mut worker_log_probs: Vec<f64> = Vec::new();
        let mut worker_rewards: Vec<f64> = Vec::new();

        let mut manager_reward_acc = 0.0;
        let mut prev_manager_state = state.clone();

        while step < self.config.max_steps_per_episode {
            // Manager: select new sub-goal every `subgoal_horizon` steps
            if subgoal_steps == 0 || subgoal_steps >= self.config.subgoal_horizon {
                if subgoal_steps > 0 {
                    manager_states.push(prev_manager_state.clone());
                    manager_subgoals.push(subgoal);
                    manager_rewards.push(manager_reward_acc);
                    manager_reward_acc = 0.0;
                }
                subgoal = self.manager.select_subgoal(&state);
                prev_manager_state = state.clone();
                subgoal_steps = 0;
            }

            // Worker: select action conditioned on state + subgoal
            let (action, log_prob) = self.worker.select_action(&state, subgoal, n_sg);
            let result = env.step(action);
            total_reward += result.reward;
            manager_reward_acc += result.reward;

            // Compute intrinsic reward (reward for making progress toward subgoal)
            let intrinsic = self.compute_intrinsic_reward(&state, &result.next_state, subgoal);

            let h_step = HierarchicalStep {
                state: state.clone(),
                subgoal,
                action,
                reward: result.reward,
                intrinsic_reward: intrinsic,
                extrinsic_reward: result.reward,
                done: result.done,
            };
            self.history.push(h_step);

            worker_states.push(state.clone());
            worker_subgoals.push(subgoal);
            worker_actions.push(action);
            worker_log_probs.push(log_prob);
            worker_rewards.push(result.reward + self.config.intrinsic_reward_scale * intrinsic);

            if result.done {
                manager_states.push(prev_manager_state.clone());
                manager_subgoals.push(subgoal);
                manager_rewards.push(manager_reward_acc);
                break;
            }

            state = result.next_state;
            step += 1;
            subgoal_steps += 1;
        }

        self.episode += 1;

        // Update manager and worker policies
        self.update_manager(&manager_states, &manager_subgoals, &manager_rewards);
        self.update_worker(
            &worker_states,
            &worker_subgoals,
            &worker_actions,
            &worker_log_probs,
            &worker_rewards,
        );

        total_reward
    }

    /// Intrinsic reward: encourage reaching sub-goal region
    /// Simple heuristic: state change magnitude weighted by subgoal direction
    fn compute_intrinsic_reward(&self, state: &[f64], next_state: &[f64], subgoal: usize) -> f64 {
        // Each subgoal corresponds to maximizing/minimizing a state dimension
        if state.is_empty() || next_state.is_empty() {
            return 0.0;
        }
        let dim = subgoal % state.len();
        let delta = next_state[dim] - state[dim];
        // Positive intrinsic reward if moving in subgoal direction
        if subgoal < state.len() {
            delta
        } else {
            -delta
        }
    }

    fn update_manager(&mut self, states: &[Vec<f64>], subgoals: &[usize], rewards: &[f64]) {
        if states.is_empty() {
            return;
        }
        let state_dim = states[0].len();
        let n = states.len();
        let n_sg = self.config.n_subgoals;

        // Compute discounted returns (manager uses coarser gamma)
        let mut returns = vec![0.0f64; n];
        let mut running = 0.0;
        for i in (0..n).rev() {
            running = rewards[i] + self.config.gamma_manager * running;
            returns[i] = running;
        }

        // Policy gradient update for manager
        let state_data: Vec<f64> = states.iter().flat_map(|s| s.iter().cloned()).collect();
        let states_t = Tensor::new(state_data, vec![n, state_dim]);

        self.manager.policy.zero_grad();
        let (probs_out, caches) = self.manager.policy.forward_with_cache(&states_t);

        let mut grad_data = vec![0.0f64; n * n_sg];
        for (i, (&sg, &ret)) in subgoals.iter().zip(returns.iter()).enumerate() {
            let p = probs_out.data[i * n_sg + sg] + 1e-10;
            grad_data[i * n_sg + sg] = -ret / (p * n as f64);
        }

        let grad_t = Tensor::new(grad_data, vec![n, n_sg]);
        self.manager.policy.backward(&grad_t, &caches);
        let mut params = self.manager.policy.collect_params();
        self.manager.optim.step(&mut params);
    }

    fn update_worker(
        &mut self,
        states: &[Vec<f64>],
        subgoals: &[usize],
        actions: &[usize],
        _log_probs: &[f64],
        rewards: &[f64],
    ) {
        if states.is_empty() {
            return;
        }
        let state_dim = states[0].len();
        let n_sg = self.config.n_subgoals;
        let n = states.len();
        let input_dim = state_dim + n_sg;
        let action_dim = self
            .worker
            .policy
            .layers
            .last()
            .map(|l| l.out_features)
            .unwrap_or(1);

        // Compute discounted returns
        let mut returns = vec![0.0f64; n];
        let mut running = 0.0;
        for i in (0..n).rev() {
            running = rewards[i] + self.config.gamma_worker * running;
            returns[i] = running;
        }

        // Build input: state concat subgoal-onehot
        let mut input_data = Vec::with_capacity(n * input_dim);
        for (s, &sg) in states.iter().zip(subgoals.iter()) {
            input_data.extend_from_slice(s);
            let mut oh = vec![0.0f64; n_sg];
            oh[sg] = 1.0;
            input_data.extend_from_slice(&oh);
        }

        let input_t = Tensor::new(input_data, vec![n, input_dim]);
        self.worker.policy.zero_grad();
        let (probs_out, caches) = self.worker.policy.forward_with_cache(&input_t);

        let mut grad_data = vec![0.0f64; n * action_dim];
        for (i, (&a, &ret)) in actions.iter().zip(returns.iter()).enumerate() {
            let p = probs_out.data[i * action_dim + a] + 1e-10;
            grad_data[i * action_dim + a] = -ret / (p * n as f64);
        }

        let grad_t = Tensor::new(grad_data, vec![n, action_dim]);
        self.worker.policy.backward(&grad_t, &caches);
        let mut params = self.worker.policy.collect_params();
        self.worker.optim.step(&mut params);
    }
}
