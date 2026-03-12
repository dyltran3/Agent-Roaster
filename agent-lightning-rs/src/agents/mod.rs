use std::sync::Arc;
use tokio::sync::Mutex;
use serde::{Serialize, Deserialize};
use thiserror::Error;
use async_trait::async_trait;

pub mod routing;
pub mod llm_agent;
pub mod roaster_optimizer;

#[derive(Error, Debug)]
pub enum AgentError {
    #[error("Execution failed: {0}")]
    ExecutionError(String),
    #[error("Handoff failed: {0}")]
    HandoffError(String),
    #[error("Validation failed: {0}")]
    ValidationError(String),
    #[error("Internal error: {0}")]
    InternalError(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResponse {
    pub content: String,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Default)]
pub struct Context {
    pub state: serde_json::Value,
    pub history: Vec<AgentResponse>,
}

#[async_trait]
pub trait Agent: Send + Sync {
    async fn execute(&self, context: Arc<Mutex<Context>>) -> Result<AgentResponse, AgentError>;
}

pub type AgentPtr = Arc<dyn Agent>;
