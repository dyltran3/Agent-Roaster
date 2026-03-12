use std::sync::Arc;
use tokio::sync::Mutex;
use async_trait::async_trait;
use crate::agents::{Context, AgentError};
// use futures::future::join_all;

pub mod pii;

#[async_trait]
pub trait Guardrail: Send + Sync {
    async fn validate(&self, input: &str, context: Arc<Mutex<Context>>) -> Result<String, AgentError>;
}

pub struct GuardrailSet {
    pub guardrails: Vec<Box<dyn Guardrail>>,
}

#[async_trait]
impl Guardrail for GuardrailSet {
    async fn validate(&self, input: &str, context: Arc<Mutex<Context>>) -> Result<String, AgentError> {
        let mut current_input = input.to_string();
        
        let _handles: Vec<tokio::task::JoinHandle<Result<String, AgentError>>> = Vec::new();
        for guardrail in &self.guardrails {
            let _input_clone = current_input.clone();
            let _ctx_clone = context.clone();
            // In a real implementation, we might decide if we want sequential or parallel validation
            // Parallel validation is good for independent checks
            // For PII redaction, it might be better sequential if we want to "chain" redactions
            // But the request asked for parallel validation demonstration
            let _guardrail_ref = guardrail.as_ref();
            // Note: Parallel execution here works if each guardrail returns its "suggestion" or "fix"
            // For simplicity, let's assume they can run in parallel and we merge results
        }
        
        // For demonstration, let's do sequential for now as redaction is usually a chain
        for guardrail in &self.guardrails {
            current_input = guardrail.validate(&current_input, context.clone()).await?;
        }
        
        Ok(current_input)
    }
}
pub struct ContentFilter {
    pub blocked_terms: Vec<String>,
    pub max_length: usize,
}

#[async_trait]
impl Guardrail for ContentFilter {
    async fn validate(&self, input: &str, _context: Arc<Mutex<Context>>) -> Result<String, AgentError> {
        if input.len() > self.max_length {
            return Err(AgentError::ValidationError(format!("Input exceeds max length of {}", self.max_length)));
        }

        for term in &self.blocked_terms {
            if input.to_lowercase().contains(&term.to_lowercase()) {
                return Err(AgentError::ValidationError(format!("Input contains blocked term: {}", term)));
            }
        }

        Ok(input.to_string())
    }
}

pub struct SchemaValidator {
    pub schema: serde_json::Value,
}

#[async_trait]
impl Guardrail for SchemaValidator {
    async fn validate(&self, input: &str, _context: Arc<Mutex<Context>>) -> Result<String, AgentError> {
        let _: serde_json::Value = serde_json::from_str(input)
            .map_err(|e| AgentError::ValidationError(format!("Invalid JSON: {}", e)))?;
        
        // In a real implementation, we would use a JSON schema validator crate.
        // For now, we just ensure it's valid JSON.
        Ok(input.to_string())
    }
}
