use super::{Agent, AgentError, AgentResponse, Context};
use crate::providers::LlmProvider;
use std::sync::Arc;
use tokio::sync::Mutex;
use async_trait::async_trait;

pub struct LlmAgent {
    pub provider: Arc<dyn LlmProvider>,
    pub prompt_template: String,
}

#[async_trait]
impl Agent for LlmAgent {
    async fn execute(&self, context: Arc<Mutex<Context>>) -> Result<AgentResponse, AgentError> {
        let history = {
            let ctx = context.lock().await;
            ctx.history.iter().map(|r| r.content.clone()).collect::<Vec<_>>().join("\n")
        };

        let full_prompt = format!("{}\n\nHistory:\n{}", self.prompt_template, history);
        let content = self.provider.completion(&full_prompt).await?;

        Ok(AgentResponse {
            content,
            metadata: serde_json::json!({ "provider": "llm" }),
        })
    }
}

pub struct LlmConditionalAgent {
    pub provider: Arc<dyn LlmProvider>,
    pub agents: Vec<(String, Arc<dyn Agent>)>,
}

#[async_trait]
impl Agent for LlmConditionalAgent {
    async fn execute(&self, context: Arc<Mutex<Context>>) -> Result<AgentResponse, AgentError> {
        let options = self.agents.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(", ");
        let prompt = format!("Based on the conversation, which agent should handle the next step? Options: [{}]", options);
        
        let decision = self.provider.completion(&prompt).await?;
        
        for (name, agent) in &self.agents {
            if decision.contains(name) {
                return agent.execute(context).await;
            }
        }

        Err(AgentError::ExecutionError(format!("LLM decision '{}' did not match any agent", decision)))
    }
}
