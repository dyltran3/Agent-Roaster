// pub mod openai;
// pub mod gemini;

use async_trait::async_trait;
use crate::agents::AgentError;

#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn completion(&self, prompt: &str) -> Result<String, AgentError>;
}
