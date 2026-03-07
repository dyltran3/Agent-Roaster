use crate::core::activation::Activation;
use crate::core::loss::value_loss;
use crate::core::optimizer::{Adam, Optimizer};
/// PPO (Proximal Policy Optimization) — Actor-Critic implementation.
///
/// Architecture:
///   - Actor network:  state → action probabilities (softmax)
///   - Critic network: state → state value scalar
///
/// Training:
///   - Collect rollouts with LightningClient
///   - Compute GAE advantages
///   - Update actor with clipped surrogate objective
///   - Update critic with MSE value loss
///   - Entropy bonus for exploration
use crate::core::tensor::Tensor;
use crate::nn::network::Sequential;
use crate::rl::buffer::RolloutBuffer;
use crate::rl::env::{Environment, POMDPStep};

pub struct PPOConfig {
    pub lr_actor: f64,
    pub lr_critic: f64,
    pub gamma: f64,
    pub lambda: f64,       // GAE smoothing factor
    pub clip_eps: f64,     // PPO clip epsilon
    pub entropy_coef: f64, // Entropy bonus coefficient
    pub value_coef: f64,   // Value loss coefficient
    pub update_epochs: usize,
    pub batch_size: usize,
    pub n_steps: usize, // Steps per rollout collection
}

impl Default for PPOConfig {
    fn default() -> Self {
        PPOConfig {
            lr_actor: 3e-4,
            lr_critic: 1e-3,
            gamma: 0.99,
            lambda: 0.95,
            clip_eps: 0.2,
            entropy_coef: 0.01,
            value_coef: 0.5,
            update_epochs: 4,
            batch_size: 64,
            n_steps: 256,
        }
    }
}

pub struct PPOAgent {
    pub actor: Sequential,
    pub critic: Sequential,
    pub actor_optim: Adam,
    pub critic_optim: Adam,
    pub config: PPOConfig,
    pub buffer: RolloutBuffer,
    pub episode: u64,
    pub step_count: u64,
}

impl PPOAgent {
    pub fn new(state_dim: usize, action_dim: usize, config: PPOConfig) -> Self {
        // Actor: state → hidden → hidden → action_probs
        let actor = Sequential::new()
            .dense(state_dim, 64, Activation::Tanh)
            .dense(64, 64, Activation::Tanh)
            .dense(64, action_dim, Activation::Softmax);

        // Critic: state → hidden → hidden → value (scalar)
        let critic = Sequential::new()
            .dense(state_dim, 64, Activation::Tanh)
            .dense(64, 64, Activation::Tanh)
            .dense(64, 1, Activation::Linear);

        let actor_optim = Adam::new(config.lr_actor);
        let critic_optim = Adam::new(config.lr_critic);

        PPOAgent {
            actor,
            critic,
            actor_optim,
            critic_optim,
            config,
            buffer: RolloutBuffer::new(),
            episode: 0,
            step_count: 0,
        }
    }

    /// Select action given current state.
    /// Returns (action, log_prob, value_estimate)
    pub fn select_action(&self, state: &[f64]) -> (usize, f64, f64) {
        let state_t = Tensor::new(state.to_vec(), vec![1, state.len()]);

        // Actor: get action probabilities
        let probs = self.actor.forward(&state_t);
        let action = sample_from_probs(&probs.data);
        let log_prob = (probs.data[action] + 1e-10).ln();

        // Critic: get value estimate
        let value = self.critic.forward(&state_t);
        let value_est = value.data[0];

        (action, log_prob, value_est)
    }

    /// Collect a rollout of n_steps from environment
    pub fn collect_rollout(&mut self, env: &mut dyn Environment) -> f64 {
        self.buffer.clear();
        let mut state = env.current_state();
        let mut total_reward = 0.0;

        for _ in 0..self.config.n_steps {
            let (action, log_prob, value_est) = self.select_action(&state);
            let result = env.step(action);
            total_reward += result.reward;

            let step = POMDPStep {
                episode_id: self.episode,
                step_id: self.step_count,
                observation: state.clone(),
                action,
                reward: result.reward,
                next_observation: result.next_state.clone(),
                done: result.done,
                log_prob,
                value_estimate: value_est,
                info: result.info,
            };

            self.buffer.push(step);
            self.step_count += 1;

            if result.done {
                state = env.reset();
                self.episode += 1;
            } else {
                state = result.next_state;
            }
        }

        self.buffer.finalize(self.config.gamma, self.config.lambda);
        total_reward
    }

    /// PPO update using a batch of episodes (transitions)
    pub fn update_from_transitions(
        &mut self,
        transitions: &[crate::rl::transition::Transition],
    ) -> (f64, f64) {
        if transitions.is_empty() {
            return (0.0, 0.0);
        }

        let mut total_actor_loss = 0.0;
        let mut total_critic_loss = 0.0;
        let n_batch = transitions.len();

        for _ in 0..self.config.update_epochs {
            for transition in transitions {
                let n_tokens = transition.rewards.len();
                if n_tokens == 0 {
                    continue;
                }

                // ── Actor Update ──────────────────────────────────────────────────
                self.actor.zero_grad();
                let (probs_out, actor_caches) = self.actor.forward_with_cache(&transition.input);
                let action_dim = probs_out.shape[1];

                let mut new_log_probs_data = Vec::with_capacity(n_tokens);
                let mut old_log_probs_data = Vec::with_capacity(n_tokens);
                let mut advantages_data = Vec::with_capacity(n_tokens);

                for i in 0..n_tokens {
                    let action_idx = transition.output.data[i] as usize;
                    let prob = probs_out.data[i * action_dim + action_idx].max(1e-10);
                    new_log_probs_data.push(prob.ln());
                    old_log_probs_data.push(transition.log_probs[i]);
                    advantages_data.push(transition.advantages[i]);
                }

                let new_lp_t = Tensor::new(new_log_probs_data, vec![n_tokens]);
                let old_lp_t = Tensor::new(old_log_probs_data, vec![n_tokens]);
                let adv_t = Tensor::new(advantages_data, vec![n_tokens]);

                let (actor_loss, grad_lp) = crate::core::loss::ppo_clip_loss(
                    &new_lp_t,
                    &old_lp_t,
                    &adv_t,
                    self.config.clip_eps,
                );

                let mut grad_actor_data = vec![0.0; n_tokens * action_dim];
                for i in 0..n_tokens {
                    let action_idx = transition.output.data[i] as usize;
                    let p = probs_out.data[i * action_dim + action_idx].max(1e-10);
                    grad_actor_data[i * action_dim + action_idx] = grad_lp.data[i] / p;
                }

                let grad_actor_t = Tensor::new(grad_actor_data, vec![n_tokens, action_dim]);
                self.actor.backward(&grad_actor_t, &actor_caches);
                let mut actor_params = self.actor.collect_params();
                self.actor_optim.step(&mut actor_params);

                total_actor_loss += actor_loss;

                // ── Critic Update ─────────────────────────────────────────────────
                // For critic, returns are advantages + old_values if using GAE,
                // but since we are in LightningRL mode with custom credit assignment,
                // we'll just treat 'advantages' as the target returns for value function.
                self.critic.zero_grad();
                let (pred_values, critic_caches) =
                    self.critic.forward_with_cache(&transition.input);

                // Target is total reward sequence or advantages (depending on context)
                // In LightningRL, step 1 gives us the target signal.
                let target_returns = Tensor::new(transition.advantages.clone(), vec![n_tokens, 1]);
                let (c_loss, c_grad) = value_loss(&pred_values, &target_returns);

                self.critic.backward(&c_grad, &critic_caches);
                let mut critic_params = self.critic.collect_params();
                self.critic_optim.step(&mut critic_params);

                total_critic_loss += c_loss;
            }
        }

        (
            total_actor_loss / (n_batch * self.config.update_epochs) as f64,
            total_critic_loss / (n_batch * self.config.update_epochs) as f64,
        )
    }

    /// Legacy update logic for local testing
    pub fn update(&mut self) -> (f64, f64) {
        // Convert RolloutBuffer items to Transitions
        let n = self.buffer.steps.len();
        if n == 0 {
            return (0.0, 0.0);
        }

        let state_dim = self.buffer.steps[0].observation.len();
        let mut input_data = Vec::with_capacity(n * state_dim);
        for s in &self.buffer.steps {
            input_data.extend_from_slice(&s.observation);
        }

        let input = Tensor::new(input_data, vec![n, state_dim]);
        let output = Tensor::new(
            self.buffer.steps.iter().map(|s| s.action as f64).collect(),
            vec![n],
        );
        let log_probs = self.buffer.steps.iter().map(|s| s.log_prob).collect();
        let rewards = self.buffer.steps.iter().map(|s| s.reward).collect();

        let mut transition =
            crate::rl::transition::Transition::new(input, output, log_probs, rewards);

        // PPO usually uses GAE
        let values: Vec<f64> = self.buffer.steps.iter().map(|s| s.value_estimate).collect();
        crate::rl::credit_assignment::assign_gae_credit(
            std::slice::from_mut(&mut transition),
            &values,
            self.config.gamma,
            self.config.lambda,
        );

        // Finalize for the transition
        let transitions = vec![transition];
        self.update_from_transitions(&transitions)
    }

    pub fn print_summary(&self) {
        println!("\n─── PPO Agent ───────────────────────────────────────");
        self.actor.print_summary();
    }
}

/// Sample action index from probability distribution
pub fn sample_from_probs(probs: &[f64]) -> usize {
    let r: f64 = crate::rand_f64();
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r <= cumulative {
            return i;
        }
    }
    probs.len() - 1
}

/// Greedy action selection (for evaluation)
pub fn greedy_action(probs: &[f64]) -> usize {
    probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}
