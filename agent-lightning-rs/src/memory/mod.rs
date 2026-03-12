use crate::agents::AgentResponse;

pub struct MemoryManager {
    pub history: Vec<AgentResponse>,
}

impl MemoryManager {
    pub fn new() -> Self {
        Self { history: Vec::new() }
    }

    pub fn add_event(&mut self, event: AgentResponse) {
        self.history.push(event);
    }

    pub async fn compact(&mut self, provider: &dyn crate::providers::LlmProvider) -> Result<(), crate::agents::AgentError> {
        if self.history.len() > 10 {
            let context = self.history.iter().map(|r| r.content.clone()).collect::<Vec<_>>().join("\n");
            let summary = provider.completion(&format!("Summarize this conversation concisely:\n\n{}", context)).await?;
            
            self.history.clear();
            self.history.push(crate::agents::AgentResponse {
                content: format!("Summary of previous context: {}", summary),
                metadata: serde_json::json!({ "type": "summary" }),
            });
        }
        Ok(())
    }
}
