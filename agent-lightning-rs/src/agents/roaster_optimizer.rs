use super::{Agent, AgentError, AgentResponse, Context};
use std::sync::Arc;
use tokio::sync::Mutex;
use async_trait::async_trait;
use serde_json::json;

pub struct RoasterOptimizer;

#[async_trait]
impl Agent for RoasterOptimizer {
    async fn execute(&self, context: Arc<Mutex<Context>>) -> Result<AgentResponse, AgentError> {
        let ctx = context.lock().await;
        
        // Extract temperature data from state
        let drum_temp = ctx.state.get("drum_temp").and_then(|t| t.as_f64()).unwrap_or(0.0);
        let bean_temp = ctx.state.get("bean_temp").and_then(|t| t.as_f64()).unwrap_or(0.0);
        let target_ror = ctx.state.get("target_ror").and_then(|r| r.as_f64()).unwrap_or(15.0);
        
        let mut burner_level = ctx.state.get("burner_level").and_then(|b| b.as_u64()).unwrap_or(50);
        
        // Simple logic for heat optimization (PID or rule-based)
        let message = if bean_temp > 200.0 {
            burner_level = burner_level.saturating_sub(10);
            "High bean temperature detected. Reducing heat.".to_string()
        } else if drum_temp < 150.0 {
            burner_level = burner_level.saturating_add(10);
            "Low drum temperature detected. Increasing heat.".to_string()
        } else {
            format!("Maintaining stable heat. Target RoR: {}°C/min", target_ror)
        };

        Ok(AgentResponse {
            content: message,
            metadata: json!({
                "action": "adjust_heat",
                "new_burner_level": burner_level,
                "timestamp": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()
            }),
        })
    }
}
