//! Model Checkpointing — save and load neural network weights (binary format).

use std::fs::File;
use std::io::{Read, Write};

/// Save a flat vector of weights to binary file
pub fn save_weights(path: &str, weights: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(path)?;
    for w in weights {
        file.write_all(&w.to_le_bytes())?;
    }
    println!("[Checkpoint] Saved {} weights to {}", weights.len(), path);
    Ok(())
}

/// Load weights from binary file
pub fn load_weights(path: &str) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    let weights: Vec<f64> = buf
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    println!(
        "[Checkpoint] Loaded {} weights from {}",
        weights.len(),
        path
    );
    Ok(weights)
}

/// Extract all weights from a PPO agent's networks
pub fn extract_ppo_weights(agent: &crate::rl::ppo::PPOAgent) -> Vec<f64> {
    let mut weights = Vec::new();
    for layer in &agent.actor.layers {
        weights.extend_from_slice(&layer.weights.data);
        weights.extend_from_slice(&layer.bias.data);
    }
    for layer in &agent.critic.layers {
        weights.extend_from_slice(&layer.weights.data);
        weights.extend_from_slice(&layer.bias.data);
    }
    weights
}
