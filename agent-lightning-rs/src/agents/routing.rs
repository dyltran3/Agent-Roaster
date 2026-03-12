use super::{Agent, AgentError, AgentResponse, Context, AgentPtr};
use std::sync::Arc;
use tokio::sync::Mutex;
use async_trait::async_trait;
use futures::future::join_all;

pub struct SequentialAgent {
    pub agents: Vec<AgentPtr>,
}

#[async_trait]
impl Agent for SequentialAgent {
    async fn execute(&self, context: Arc<Mutex<Context>>) -> Result<AgentResponse, AgentError> {
        let mut last_response = AgentResponse {
            content: "Start of sequential execution".to_string(),
            metadata: serde_json::json!({}),
        };

        for agent in &self.agents {
            last_response = agent.execute(context.clone()).await?;
            context.lock().await.history.push(last_response.clone());
        }

        Ok(last_response)
    }
}

pub struct ParallelAgent {
    pub agents: Vec<AgentPtr>,
}

#[async_trait]
impl Agent for ParallelAgent {
    async fn execute(&self, context: Arc<Mutex<Context>>) -> Result<AgentResponse, AgentError> {
        let mut handles = Vec::new();

        for agent in &self.agents {
            let ctx = context.clone();
            let agent_clone = agent.clone();
            let handle = tokio::spawn(async move {
                agent_clone.execute(ctx).await
            });
            handles.push(handle);
        }

        let results = join_all(handles).await;
        let mut contents = Vec::new();
        let mut metadata = serde_json::Map::new();

        for (i, result) in results.into_iter().enumerate() {
            match result {
                Ok(Ok(response)) => {
                    contents.push(response.content);
                    metadata.insert(format!("agent_{}", i), response.metadata);
                }
                Ok(Err(e)) => return Err(e),
                Err(e) => return Err(AgentError::InternalError(e.to_string())),
            }
        }

        Ok(AgentResponse {
            content: contents.join("\n---\n"),
            metadata: serde_json::Value::Object(metadata),
        })
    }
}

pub struct ConditionalAgent<F>
where
    F: Fn(&Context) -> usize + Send + Sync,
{
    pub agents: Vec<AgentPtr>,
    pub selector: F,
}

#[async_trait]
impl<F> Agent for ConditionalAgent<F>
where
    F: Fn(&Context) -> usize + Send + Sync,
{
    async fn execute(&self, context: Arc<Mutex<Context>>) -> Result<AgentResponse, AgentError> {
        let index = {
            let ctx = context.lock().await;
            (self.selector)(&*ctx)
        };

        if index < self.agents.len() {
            self.agents[index].execute(context).await
        } else {
            Err(AgentError::ExecutionError("Selector index out of bounds".to_string()))
        }
    }
}

pub struct LoopAgent {
    pub agent: AgentPtr,
    pub condition: Box<dyn Fn(&Context) -> bool + Send + Sync>,
}

#[async_trait]
impl Agent for LoopAgent {
    async fn execute(&self, context: Arc<Mutex<Context>>) -> Result<AgentResponse, AgentError> {
        let resp = self.agent.execute(context.clone()).await?;
        let mut last_response = resp;

        loop {
            let ctx = context.lock().await;
            if !(self.condition)(&*ctx) {
                break;
            }
            last_response = self.agent.execute(context.clone()).await?;
        }

        Ok(last_response)
    }
}

pub struct CustomAgent<F>
where
    F: Fn(Arc<Mutex<Context>>) -> futures::future::BoxFuture<'static, Result<AgentResponse, AgentError>> + Send + Sync,
{
    pub func: F,
}

#[async_trait]
impl<F> Agent for CustomAgent<F>
where
    F: Fn(Arc<Mutex<Context>>) -> futures::future::BoxFuture<'static, Result<AgentResponse, AgentError>> + Send + Sync,
{
    async fn execute(&self, context: Arc<Mutex<Context>>) -> Result<AgentResponse, AgentError> {
        (self.func)(context).await
    }
}
