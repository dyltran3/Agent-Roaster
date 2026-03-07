use crate::core::activation::Activation;
use crate::core::optimizer::Adam;
use crate::core::optimizer::Optimizer;
/// GRPO (Group Relative Policy Optimization)
///
/// Core idea: instead of computing per-step advantages with a critic,
/// sample G trajectories for same state, compute relative rewards within group,
/// normalize, and use as advantages. No Value network needed.
///
/// Reference: DeepSeek-R1 / Microsoft Agent Lightning compatible.
use crate::core::tensor::Tensor;
use crate::nn::network::Sequential;
use crate::rl::env::Environment;
use crate::rl::ppo::sample_from_probs;

pub struct GRPOConfig {
    pub lr: f64,
    pub gamma: f64,
    pub group_size: usize, // G: number of trajectories per group
    pub clip_eps: f64,
    pub kl_coef: f64, // KL divergence penalty coefficient
    pub n_episodes_per_update: usize,
    pub max_steps: usize, // Max steps per episode
}

impl Default for GRPOConfig {
    fn default() -> Self {
        GRPOConfig {
            lr: 3e-4,
            gamma: 0.99,
            group_size: 8,
            clip_eps: 0.2,
            kl_coef: 0.01,
            n_episodes_per_update: 4,
            max_steps: 200,
        }
    }
}

pub struct GRPOAgent {
    pub policy: Sequential,
    pub old_policy_log_probs: Vec<Vec<f64>>, // Cached old policy probs
    pub optim: Adam,
    pub config: GRPOConfig,
    pub episode: u64,
}

impl GRPOAgent {
    pub fn new(state_dim: usize, action_dim: usize, config: GRPOConfig) -> Self {
        let policy = Sequential::new()
            .dense(state_dim, 128, Activation::Tanh)
            .dense(128, 64, Activation::Tanh)
            .dense(64, action_dim, Activation::Softmax);

        let optim = Adam::new(config.lr);

        GRPOAgent {
            policy,
            old_policy_log_probs: Vec::new(),
            optim,
            config,
            episode: 0,
        }
    }

    /// Roll out one full episode, collecting (state, action, log_prob, reward)
    fn rollout_episode(
        &self,
        env: &mut dyn Environment,
    ) -> (Vec<Vec<f64>>, Vec<usize>, Vec<f64>, f64) {
        let mut states = Vec::new();
        let mut actions = Vec::new();
        let mut log_probs = Vec::new();
        let mut total_reward = 0.0;

        let mut state = env.reset();
        for _ in 0..self.config.max_steps {
            let state_t = Tensor::new(state.clone(), vec![1, state.len()]);
            let probs = self.policy.forward(&state_t);
            let action = sample_from_probs(&probs.data);
            let log_prob = (probs.data[action] + 1e-10).ln();

            let result = env.step(action);
            total_reward += result.reward;

            states.push(state.clone());
            actions.push(action);
            log_probs.push(log_prob);

            if result.done {
                break;
            }
            state = result.next_state;
        }
        (states, actions, log_probs, total_reward)
    }

    /// GRPO update using a batch of episodes (transitions)
    /// This reflects the "Server" side of Agent Lightning.
    pub fn update_from_transitions(
        &mut self,
        transitions: &[crate::rl::transition::Transition],
    ) -> f64 {
        if transitions.is_empty() {
            return 0.0;
        }

        let mut total_loss = 0.0;
        let n_batch = transitions.len();

        for transition in transitions {
            let n_tokens = transition.rewards.len();
            if n_tokens == 0 {
                continue;
            }

            self.policy.zero_grad();

            // Forward pass for this transition's input
            let (probs_out, caches) = self.policy.forward_with_cache(&transition.input);
            let action_dim = probs_out.shape[1];

            // In token-level RL, each token in the output sequence is an action.
            // But if 'output' is a sequence, probs_out should be [seq_len, action_dim].
            // If the policy is a simple feed-forward, we might need to expand or handle sequence data.
            // For now, assume transition.output contains indices and probs_out is batch-formatted.

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

            // Step 2: Token-level Policy Optimization
            let (loss_val, grad_lp) =
                crate::core::loss::grpo_loss(&new_lp_t, &old_lp_t, &adv_t, self.config.clip_eps);

            // Convert dL/d_log_prob to dL/d_logits/probs for policy network
            let mut grad_policy_data = vec![0.0; n_tokens * action_dim];
            for i in 0..n_tokens {
                let action_idx = transition.output.data[i] as usize;
                let p = probs_out.data[i * action_dim + action_idx].max(1e-10);
                // dL/dp = (dL/d_lp) * (1/p)
                grad_policy_data[i * action_dim + action_idx] = grad_lp.data[i] / p;
            }

            let grad_policy_t = Tensor::new(grad_policy_data, vec![n_tokens, action_dim]);
            self.policy.backward(&grad_policy_t, &caches);

            let mut params = self.policy.collect_params();
            self.optim.step(&mut params);

            total_loss += loss_val;
        }

        total_loss / n_batch as f64
    }

    /// Legend rollout logic kept for local testing/debugging
    pub fn update(&mut self, env: &mut dyn Environment) -> f64 {
        let g = self.config.group_size;
        let mut group_rewards = Vec::with_capacity(g);
        let mut group_transitions = Vec::with_capacity(g);

        // Collect G rollouts
        for _ in 0..g {
            let (states, actions, log_probs, episode_reward) = self.rollout_episode(env);
            group_rewards.push(episode_reward);

            // Convert to Transition structure
            let n = states.len();
            if n == 0 {
                continue;
            }

            let state_dim = states[0].len();
            let mut input_data = Vec::with_capacity(n * state_dim);
            for s in states {
                input_data.extend(s);
            }

            let input = Tensor::new(input_data, vec![n, state_dim]);
            let output = Tensor::new(actions.iter().map(|&a| a as f64).collect(), vec![n]);

            // In this simple case, reward is 0 for all internal steps, and episode_reward at the end.
            let mut per_step_rewards = vec![0.0; n];
            per_step_rewards[n - 1] = episode_reward;

            group_transitions.push(crate::rl::transition::Transition::new(
                input,
                output,
                log_probs,
                per_step_rewards,
            ));
        }
        self.episode += g as u64;

        // Step 1: Credit Assignment (Group Relative)
        let mean_r: f64 = group_rewards.iter().sum::<f64>() / g as f64;
        let std_r: f64 = {
            let var = group_rewards
                .iter()
                .map(|r| (r - mean_r).powi(2))
                .sum::<f64>()
                / g as f64;
            (var + 1e-8).sqrt()
        };

        for transition in group_transitions.iter_mut() {
            let total_r = transition.total_reward();
            let advantage = (total_r - mean_r) / std_r;
            let n = transition.rewards.len();
            transition.advantages = vec![advantage; n]; // Uniform across sequence
        }

        // Step 2: Update from transitions
        self.update_from_transitions(&group_transitions)
    }

    pub fn select_action(&self, state: &[f64]) -> usize {
        let state_t = Tensor::new(state.to_vec(), vec![1, state.len()]);
        let probs = self.policy.forward(&state_t);
        sample_from_probs(&probs.data)
    }
}
