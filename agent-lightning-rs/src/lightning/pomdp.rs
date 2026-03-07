//! POMDP State Abstraction — the universal interface between agent and training.
//! (serde derives removed to avoid proc-macro dependency on GNU toolchain)

/// A single observation from a partially observable world
#[derive(Debug, Clone)]
pub struct Observation {
    pub raw: Vec<f64>,
    pub metadata: std::collections::HashMap<String, String>,
}

impl Observation {
    pub fn new(raw: Vec<f64>) -> Self {
        Observation {
            raw,
            metadata: std::collections::HashMap::new(),
        }
    }

    pub fn with_meta(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

/// An agent event — the atomic unit sent from Lightning Client to Server
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// Agent took an action
    Action {
        obs: Observation,
        action_id: usize,
        action_repr: String,
        log_prob: f64,
    },
    /// Agent received a reward signal
    Reward { value: f64, source: RewardSource },
    /// Agent encountered an error
    Error { message: String, recoverable: bool },
    /// Episode/task boundary
    TaskBoundary { success: bool, total_reward: f64 },
}

/// Where the reward came from
#[derive(Debug, Clone)]
pub enum RewardSource {
    Environment,
    Intermediate,
    ErrorPenalty,
    UserFeedback,
    ToolSuccess,
}

/// POMDP transition tuple
#[derive(Debug, Clone)]
pub struct POMDPTransition {
    pub state: Observation,
    pub action: usize,
    pub reward: f64,                    // Terminal reward
    pub intermediate_rewards: Vec<f64>, // AIR rewards
    pub next_state: Observation,
    pub done: bool,
    pub reward_source: RewardSource,
    pub error: Option<String>,
}
