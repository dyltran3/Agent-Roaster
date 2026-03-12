// use std::sync::Arc;
use tokio::sync::mpsc;
use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use thiserror::Error;

pub mod control_panel;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AudioFormat {
    Pcm16,
    G711,
    Opus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFrame {
    pub data: Vec<u8>,
    pub format: AudioFormat,
    pub timestamp: u64,
}

#[derive(Error, Debug)]
pub enum TransportError {
    #[error("Connection failed: {0}")]
    ConnectionError(String),
    #[error("Send failed: {0}")]
    SendError(String),
    #[error("Receive failed: {0}")]
    ReceiveError(String),
}

#[async_trait]
pub trait Transport: Send + Sync {
    async fn connect(&mut self, url: &str) -> Result<(), TransportError>;
    async fn send_frame(&self, frame: AudioFrame) -> Result<(), TransportError>;
    async fn set_receiver(&mut self, tx: mpsc::Sender<AudioFrame>);
}

#[async_trait]
pub trait AudioStream: Send + Sync {
    async fn start(&mut self) -> Result<(), TransportError>;
    async fn stop(&mut self) -> Result<(), TransportError>;
}
