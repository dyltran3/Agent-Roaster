/// Training configuration — set programmatically or via code defaults.
/// (TOML loading removed to avoid serde proc-macro dependency on GNU toolchain)

#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub algorithm: String,       // "ppo" | "grpo" | "hierarchical"
    pub environment: String,     // "gridworld" | "cartpole"
    pub total_episodes: usize,
    pub log_every: usize,
    pub eval_every: usize,
    pub eval_episodes: usize,
    pub seed: u64,
    pub verbose: bool,

    // PPO specific
    pub lr_actor: f64,
    pub lr_critic: f64,
    pub gamma: f64,
    pub lambda: f64,
    pub clip_eps: f64,
    pub entropy_coef: f64,
    pub update_epochs: usize,
    pub batch_size: usize,
    pub n_steps: usize,

    // GRPO specific
    pub group_size: usize,
    pub kl_coef: f64,

    // Hierarchical specific
    pub n_subgoals: usize,
    pub subgoal_horizon: usize,
    pub lr_manager: f64,
    pub lr_worker: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        TrainingConfig {
            algorithm: "ppo".to_string(),
            environment: "gridworld".to_string(),
            total_episodes: 500,
            log_every: 10,
            eval_every: 50,
            eval_episodes: 5,
            seed: 42,
            verbose: true,

            lr_actor: 3e-4,
            lr_critic: 1e-3,
            gamma: 0.99,
            lambda: 0.95,
            clip_eps: 0.2,
            entropy_coef: 0.01,
            update_epochs: 4,
            batch_size: 64,
            n_steps: 256,

            group_size: 8,
            kl_coef: 0.01,

            n_subgoals: 4,
            subgoal_horizon: 10,
            lr_manager: 1e-4,
            lr_worker: 3e-4,
        }
    }
}
