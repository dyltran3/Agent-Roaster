use std::sync::Arc;
use tokio::sync::Mutex;
use crate::agents::{Context, AgentError};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlCommand {
    pub target: String,
    pub value: serde_json::Value,
}

pub struct ControlPanel {
    pub context: Arc<Mutex<Context>>,
}

impl ControlPanel {
    pub fn new(context: Arc<Mutex<Context>>) -> Self {
        Self { context }
    }

    pub async fn apply_override(&self, command: ControlCommand) -> Result<(), AgentError> {
        let mut ctx = self.context.lock().await;
        
        // Apply manual override to context state
        match ctx.state.as_object_mut() {
            Some(obj) => {
                obj.insert(command.target, command.value);
                Ok(())
            }
            None => Err(AgentError::InternalError("Context state is not an object".to_string())),
        }
    }

    pub async fn get_status(&self) -> serde_json::Value {
        let ctx = self.context.lock().await;
        serde_json::to_value(&ctx.state).unwrap_or(serde_json::json!({}))
    }
}
