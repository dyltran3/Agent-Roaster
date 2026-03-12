use super::Guardrail;
use crate::agents::{Context, AgentError};
use std::sync::Arc;
use tokio::sync::Mutex;
use async_trait::async_trait;
use regex::Regex;

pub struct PiiRedactor {
    pub patterns: Vec<(String, Regex)>,
}

impl Default for PiiRedactor {
    fn default() -> Self {
        Self {
            patterns: vec![
                ("EMAIL".to_string(), Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}").unwrap()),
                ("PHONE".to_string(), Regex::new(r"\+?\d{1,4}?[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}").unwrap()),
                ("IP".to_string(), Regex::new(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b").unwrap()),
                ("CC".to_string(), Regex::new(r"\b(?:\d[ -]*?){13,16}\b").unwrap()),
            ],
        }
    }
}

#[async_trait]
impl Guardrail for PiiRedactor {
    async fn validate(&self, input: &str, _context: Arc<Mutex<Context>>) -> Result<String, AgentError> {
        let mut redacted = input.to_string();
        for (label, regex) in &self.patterns {
            redacted = regex.replace_all(&redacted, format!("[REDACTED {}]", label)).to_string();
        }
        Ok(redacted)
    }
}
