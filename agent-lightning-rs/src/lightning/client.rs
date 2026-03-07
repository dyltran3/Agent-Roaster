use crate::lightning_log;
use crate::rl::transition::Transition;
/// Lightning Client — co-exists with agent, traces execution and sends data to server.
///
/// Uses std::sync::mpsc (zero external dependencies, no proc-macros).
/// The client is the "sensor" side of Agent Lightning's Training-Agent Disaggregation.
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};

// ─── Channel Messages ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum ClientToServerMsg {
    Transition(Transition),
    Error {
        episode_id: u64,
        step_id: u64,
        description: String,
    },
    EpisodeDone {
        episode_id: u64,
        total_reward: f64,
        intermediate_rewards: Vec<f64>,
    },
}

#[derive(Debug, Clone)]
pub enum ServerToClientMsg {
    PolicyUpdate { params: Vec<f64>, version: u64 },
    Pause,
    Resume,
}

// ─── Lightning Client ─────────────────────────────────────────────────────────

pub struct LightningClient {
    pub client_id: u64,
    sender: Sender<ClientToServerMsg>,
    receiver: Arc<Mutex<Receiver<ServerToClientMsg>>>,
    pub policy_version: u64,
    pub step_count: u64,
    pub episode_id: u64,
    paused: bool,
}

impl LightningClient {
    pub fn new(
        client_id: u64,
        sender: Sender<ClientToServerMsg>,
        server_rx: Receiver<ServerToClientMsg>,
    ) -> Self {
        LightningClient {
            client_id,
            sender,
            receiver: Arc::new(Mutex::new(server_rx)),
            policy_version: 0,
            step_count: 0,
            episode_id: 0,
            paused: false,
        }
    }

    /// Record one agent reasoning/action transition and send to training server
    pub fn trace_transition(&mut self, transition: Transition) {
        if self.paused {
            return;
        }
        self.step_count += 1;
        let _ = self.sender.send(ClientToServerMsg::Transition(transition));
        self.poll_server_messages();
    }

    /// Record an agent error for recovery learning
    pub fn record_error(&mut self, description: String) {
        let _ = self.sender.send(ClientToServerMsg::Error {
            episode_id: self.episode_id,
            step_id: self.step_count,
            description,
        });
    }

    /// Signal episode completion with optional intermediate reward signals
    pub fn episode_done(&mut self, total_reward: f64, intermediate_rewards: Vec<f64>) {
        self.episode_id += 1;
        let _ = self.sender.send(ClientToServerMsg::EpisodeDone {
            episode_id: self.episode_id - 1,
            total_reward,
            intermediate_rewards,
        });
    }

    /// Non-blocking poll for server messages (policy updates, pause/resume)
    pub fn poll_server_messages(&mut self) {
        if let Ok(rx) = self.receiver.try_lock() {
            loop {
                match rx.try_recv() {
                    Ok(msg) => match msg {
                        ServerToClientMsg::PolicyUpdate { params, version } => {
                            lightning_log!(
                                info,
                                "[Client {}] Policy update v{} ({} params)",
                                self.client_id,
                                version,
                                params.len()
                            );
                            self.policy_version = version;
                        }
                        ServerToClientMsg::Pause => {
                            self.paused = true;
                        }
                        ServerToClientMsg::Resume => {
                            self.paused = false;
                        }
                    },
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => break,
                }
            }
        }
    }

    pub fn current_policy_version(&self) -> u64 {
        self.policy_version
    }
}

// ─── Channel factory ──────────────────────────────────────────────────────────

pub struct ClientServerPair {
    pub client_tx: Sender<ClientToServerMsg>,
    pub client_rx: Receiver<ClientToServerMsg>,
    pub server_tx: Sender<ServerToClientMsg>,
    pub server_rx: Receiver<ServerToClientMsg>,
}

pub fn create_client_server_channels() -> ClientServerPair {
    let (client_tx, client_rx) = mpsc::channel();
    let (server_tx, server_rx) = mpsc::channel();
    ClientServerPair {
        client_tx,
        client_rx,
        server_tx,
        server_rx,
    }
}
