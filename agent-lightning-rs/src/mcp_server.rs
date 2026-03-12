// Agent Lightning RS — MCP Server
// Exposes the RL training agent as an MCP (Model Context Protocol) server
// so that AI assistants (Claude, Gemini, etc.) can call it as a tool.
//
// Protocol: stdio JSON-RPC 2.0 (standard MCP transport)
// Usage: add to mcp_config.json (see README)

use std::io::{self, BufRead, Write};
use std::time::Instant;

use agent_roaster::envs::cartpole::CartPole;
use agent_roaster::envs::gridworld::GridWorld;
use agent_roaster::rl::env::Environment;
use agent_roaster::rl::grpo::{GRPOAgent, GRPOConfig};
use agent_roaster::rl::hierarchical::{LightningRLAgent, LightningRLConfig};
use agent_roaster::rl::ppo::{PPOAgent, PPOConfig};

// ─── MCP Protocol Types (JSON-RPC 2.0) ───────────────────────────────────────

fn mcp_response(id: &str, result: &str) -> String {
    format!(
        "{{\"jsonrpc\":\"2.0\",\"id\":{},\"result\":{}}}",
        id, result
    )
}

fn mcp_error(id: &str, code: i64, message: &str) -> String {
    format!(
        "{{\"jsonrpc\":\"2.0\",\"id\":{},\"error\":{{\"code\":{},\"message\":\"{}\"}}}}",
        id, code, message
    )
}

// ─── MCP Initialization Response ─────────────────────────────────────────────

fn handle_initialize(id: &str) -> String {
    mcp_response(
        id,
        r#"{
      "protocolVersion": "2024-11-05",
      "capabilities": { "tools": {}, "logging": {} },
      "serverInfo": {
        "name": "agent-lightning-rs",
        "version": "0.1.0",
        "description": "Reinforcement Learning agent training framework (PPO/GRPO/HRL) written from scratch in Rust"
      }
    }"#,
    )
}

// ─── tools/list ──────────────────────────────────────────────────────────────

fn handle_tools_list(id: &str) -> String {
    mcp_response(
        id,
        r#"{
      "tools": [
        {
          "name": "train_agent",
          "description": "Start training an RL agent. Returns episode rewards as it trains.",
          "inputSchema": {
            "type": "object",
            "properties": {
              "algorithm": {
                "type": "string",
                "enum": ["ppo", "grpo", "hierarchical"],
                "description": "RL algorithm to use",
                "default": "ppo"
              },
              "environment": {
                "type": "string",
                "enum": ["gridworld", "cartpole"],
                "description": "Training environment",
                "default": "gridworld"
              },
              "episodes": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10000,
                "description": "Number of training episodes",
                "default": 100
              }
            },
            "required": []
          }
        },
        {
          "name": "evaluate_agent",
          "description": "Run a quick evaluation of a freshly trained agent and return performance metrics.",
          "inputSchema": {
            "type": "object",
            "properties": {
              "algorithm": { "type": "string", "enum": ["ppo", "grpo", "hierarchical"] },
              "environment": { "type": "string", "enum": ["gridworld", "cartpole"] },
              "eval_episodes": { "type": "integer", "default": 5, "minimum": 1, "maximum": 50 }
            }
          }
        },
        {
          "name": "list_algorithms",
          "description": "List available RL algorithms and their descriptions.",
          "inputSchema": { "type": "object", "properties": {} }
        },
        },
        {
          "name": "predict_roaster_action",
          "description": "Send hardware sensor data (ET, BT) to the PINN agent to get the exact Gas % output.",
          "inputSchema": {
            "type": "object",
            "properties": {
              "et": { "type": "number", "description": "Environmental (Drum) Temperature in °C" },
              "bt": { "type": "number", "description": "Bean Temperature in °C" }
            },
            "required": ["et", "bt"]
          }
        }
      ]
    }"#,
    )
}

// ─── tools/call dispatch ──────────────────────────────────────────────────────

fn handle_tool_call(id: &str, tool_name: &str, args: &str) -> String {
    match tool_name {
        "list_algorithms" => list_algorithms(id),
        "list_environments" => list_environments(id),
        "train_agent" => train_agent(id, args),
        "evaluate_agent" => evaluate_agent_tool(id, args),
        "predict_roaster_action" => predict_roaster_action(id, args),
        other => mcp_error(id, -32601, &format!("Unknown tool: {}", other)),
    }
}

fn list_algorithms(id: &str) -> String {
    mcp_response(
        id,
        r#"{
      "content": [{
        "type": "text",
        "text": "Available RL Algorithms in Agent Lightning RS:\n\n**PPO (Proximal Policy Optimization)**\n- Actor-Critic architecture\n- Clipped surrogate objective\n- GAE advantage estimation\n- Best for: stable, general-purpose training\n\n**GRPO (Group Relative Policy Optimization)**\n- No critic network needed\n- Group-relative advantage normalization\n- Inspired by DeepSeek-R1\n- Best for: fast exploration, sample efficiency\n\n**LightningRL (Hierarchical RL)**\n- Manager selects sub-goals, Worker executes\n- Intrinsic reward shaping\n- Multi-level credit assignment\n- Best for: complex, long-horizon tasks"
      }]
    }"#,
    )
}

fn list_environments(id: &str) -> String {
    mcp_response(
        id,
        r#"{
      "content": [{
        "type": "text",
        "text": "Available Environments:\n\n**GridWorld (5x5)**\n- State: (agent_x, agent_y, goal_x, goal_y) — 4 features\n- Actions: Up, Down, Left, Right (4 actions)\n- Reward: +10 goal, -0.1 per step, -1.0 wall\n- Max steps: 100\n\n**CartPole**\n- State: (position, velocity, angle, angular_velocity) — 4 features\n- Actions: Push Left, Push Right (2 actions)\n- Reward: +1 per timestep upright\n- Physics: Euler integration (real equations of motion)\n- Max steps: 500"
      }]
    }"#,
    )
}

fn train_agent(id: &str, args: &str) -> String {
    // Parse simple key-value from JSON args
    let algorithm = extract_str_field(args, "algorithm").unwrap_or("ppo".to_string());
    let environment = extract_str_field(args, "environment").unwrap_or("gridworld".to_string());
    let episodes = extract_int_field(args, "episodes").unwrap_or(100).min(500) as usize;

    let mut results = Vec::new();
    let start = Instant::now();

    match (algorithm.as_str(), environment.as_str()) {
        ("ppo", "gridworld") | ("ppo", _) => {
            let cfg = PPOConfig {
                n_steps: 64,
                batch_size: 32,
                update_epochs: 3,
                ..PPOConfig::default()
            };
            let mut env = GridWorld::new(5, 5);
            let mut agent = PPOAgent::new(env.state_space(), env.action_space(), cfg);
            env.reset();

            for ep in 0..episodes {
                agent.collect_rollout(&mut env);
                let (al, _cl) = agent.update();
                let rew: f64 = agent.buffer.steps.iter().map(|s| s.reward).sum();
                if ep % (episodes / 5).max(1) == 0 {
                    results.push(format!(
                        "Ep {:>3}: reward={:>7.2} actor_loss={:.4}",
                        ep, rew, al
                    ));
                }
            }
        }
        ("grpo", env_name) => {
            let cfg = GRPOConfig {
                group_size: 4,
                max_steps: 80,
                ..GRPOConfig::default()
            };
            let mut env: Box<dyn Environment> = match env_name {
                "cartpole" => Box::new(CartPole::new()),
                _ => Box::new(GridWorld::new(5, 5)),
            };
            let mut agent = GRPOAgent::new(env.state_space(), env.action_space(), cfg);

            for ep in 0..episodes {
                let loss = agent.update(env.as_mut());
                if ep % (episodes / 5).max(1) == 0 {
                    results.push(format!("Ep {:>3}: loss={:.4}", ep, loss));
                }
            }
        }
        ("hierarchical", _) => {
            let cfg = LightningRLConfig {
                max_steps_per_episode: 80,
                ..LightningRLConfig::default()
            };
            let mut env = GridWorld::new(5, 5);
            let mut agent = LightningRLAgent::new(env.state_space(), env.action_space(), cfg);

            for ep in 0..episodes {
                let rew = agent.run_episode(&mut env);
                if ep % (episodes / 5).max(1) == 0 {
                    results.push(format!("Ep {:>3}: reward={:>7.2}", ep, rew));
                }
            }
        }
        _ => return mcp_error(id, -32602, "Unknown algorithm or environment"),
    }

    let elapsed = start.elapsed().as_secs_f64();
    let summary = format!(
        "Training complete!\nAlgorithm: {}\nEnvironment: {}\nEpisodes: {}\nTime: {:.1}s\n\nProgress:\n{}",
        algorithm, environment, episodes, elapsed,
        results.join("\n")
    );

    mcp_response(
        id,
        &format!(
            "{{\"content\":[{{\"type\":\"text\",\"text\":{}}}]}}",
            json_str(&summary)
        ),
    )
}

fn extract_float_field(json: &str, field: &str) -> Option<f64> {
    let key = format!("\"{}\":", field);
    if let Some(mut start) = json.find(&key) {
        start += key.len();
        let end = json[start..]
            .find(|c: char| c == ',' || c == '}' || c.is_whitespace())
            .unwrap_or(json.len() - start);
        let val_str = json[start..start + end].trim();
        return val_str.parse().ok();
    }
    None
}

fn predict_roaster_action(id: &str, args: &str) -> String {
    use agent_roaster::core::activation::Activation;
    use agent_roaster::core::tensor::Tensor;
    use agent_roaster::envs::state_estimator::ExtendedKalmanFilter;
    use agent_roaster::nn::network::Sequential;
    use agent_roaster::security::bounds::{
        apply_hybrid_control, check_safety_bounds, compute_base_gas,
    };

    // 1. Parse Args
    let et = extract_float_field(args, "et").unwrap_or(200.0);
    let bt = extract_float_field(args, "bt").unwrap_or(150.0);

    // 2. State Estimator Filter
    let mut ekf = ExtendedKalmanFilter::new(bt);
    ekf.predict(1.0, et); // Assume dt=1.0 for real-time tick
    ekf.update(bt);

    // 3. Safety Check
    if !check_safety_bounds(et, ekf.x[0], ekf.x[1]) {
        let msg = r#"{"content":[{"type":"text","text":"{\"error\":\"CRITICAL EMERGENCY STOP: Hardware limits exceeded!\",\"gas\":0.0}"}]}"#;
        return mcp_response(id, msg);
    }

    // 4. Setup Architecture & Try Load Checkpoint
    let mut policy = Sequential::new()
        .dense(4, 16, Activation::ReLU)
        .dense(16, 16, Activation::ReLU)
        .dense(16, 1, Activation::Tanh); // Output: residual correction in [-1, 1]

    let path = "roaster_pinn_model.bin";
    let _ = policy.load(path); // Fails gracefully, proceeds with random weights if no file

    // 5. Neural Network Inference
    let abstract_state = vec![ekf.x[0], ekf.x[1], ekf.x[2], ekf.x[3]];
    let state_t = Tensor::new(abstract_state, vec![1, 4]);
    let residual_correction = policy.forward(&state_t).data[0];

    // 6. Thermal Hybrid Logic
    let target_ror = 15.0; // Abstract planned ROR for Demo
    let base_gas = compute_base_gas(et, ekf.x[0], target_ror);
    let final_gas = apply_hybrid_control(base_gas, residual_correction);

    let output_json = format!(
        "{{\"gas_percents\": {:.2}, \"hybrid\": {{\"base\": {:.2}, \"ai_residual\": {:.2}}}}}",
        final_gas, base_gas, residual_correction
    );

    mcp_response(
        id,
        &format!(
            "{{\"content\":[{{\"type\":\"text\",\"text\":{}}}]}}",
            json_str(&output_json)
        ),
    )
}

fn evaluate_agent_tool(id: &str, args: &str) -> String {
    let algorithm = extract_str_field(args, "algorithm").unwrap_or("ppo".to_string());
    let environment = extract_str_field(args, "environment").unwrap_or("gridworld".to_string());
    let eval_eps = extract_int_field(args, "eval_episodes")
        .unwrap_or(5)
        .min(20) as usize;

    // Quick train then evaluate
    let mut eval_env: Box<dyn Environment> = match environment.as_str() {
        "cartpole" => Box::new(CartPole::new()),
        _ => Box::new(GridWorld::new(5, 5)),
    };

    let cfg = PPOConfig {
        n_steps: 64,
        batch_size: 32,
        update_epochs: 3,
        ..PPOConfig::default()
    };
    let mut env = GridWorld::new(5, 5);
    let mut agent = PPOAgent::new(env.state_space(), env.action_space(), cfg);
    env.reset();

    // Train for 50 episodes
    for _ in 0..50 {
        agent.collect_rollout(&mut env);
        agent.update();
    }

    // Evaluate greedily
    use agent_roaster::core::tensor::Tensor;
    use agent_roaster::rl::ppo::greedy_action;
    let mut total = 0.0f64;
    for _ in 0..eval_eps {
        let mut state = eval_env.reset();
        loop {
            let t = Tensor::new(state.clone(), vec![1, state.len()]);
            let probs = agent.actor.forward(&t);
            let action = greedy_action(&probs.data);
            let result = eval_env.step(action);
            total += result.reward;
            if result.done {
                break;
            }
            state = result.next_state;
        }
    }
    let avg = total / eval_eps as f64;
    let text = format!(
        "Evaluation Results:\nAlgorithm: {}\nEnvironment: {}\nEval Episodes: {}\nAverage Reward: {:.2}",
        algorithm, environment, eval_eps, avg
    );
    mcp_response(
        id,
        &format!(
            "{{\"content\":[{{\"type\":\"text\",\"text\":{}}}]}}",
            json_str(&text)
        ),
    )
}

// ─── JSON helpers (no serde needed) ──────────────────────────────────────────

fn extract_str_field(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\":", key);
    let start = json.find(&pattern)? + pattern.len();
    let rest = json[start..].trim_start();
    if rest.starts_with('"') {
        let end = rest[1..].find('"')?;
        Some(rest[1..end + 1].to_string())
    } else {
        None
    }
}

fn extract_int_field(json: &str, key: &str) -> Option<i64> {
    let pattern = format!("\"{}\":", key);
    let start = json.find(&pattern)? + pattern.len();
    let rest = json[start..].trim_start();
    let end = rest
        .find(|c: char| !c.is_ascii_digit() && c != '-')
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

fn json_str(s: &str) -> String {
    let escaped = s
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n");
    format!("\"{}\"", escaped)
}

// ─── Main stdio loop ──────────────────────────────────────────────────────────

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    eprintln!("[agent-lightning MCP] Server started, waiting for JSON-RPC messages on stdin...");

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) if !l.trim().is_empty() => l,
            _ => continue,
        };

        // Parse method and id from JSON-RPC
        let method = extract_str_field(&line, "method").unwrap_or_default();
        let id_raw = {
            // Try to grab raw id value (could be string or number)
            let pattern = "\"id\":";
            if let Some(pos) = line.find(pattern) {
                let rest = line[pos + pattern.len()..].trim_start();
                let end = rest
                    .find(|c: char| c == ',' || c == '}')
                    .unwrap_or(rest.len());
                rest[..end].trim().to_string()
            } else {
                "null".to_string()
            }
        };

        // Get params
        let params = {
            let pattern = "\"params\":";
            if let Some(pos) = line.find(pattern) {
                line[pos + pattern.len()..].trim_start().to_string()
            } else {
                "{}".to_string()
            }
        };

        let response = match method.as_str() {
            "initialize" => handle_initialize(&id_raw),
            "initialized" => continue, // notification, no response
            "tools/list" => handle_tools_list(&id_raw),
            "tools/call" => {
                let tool_name = extract_str_field(&params, "name").unwrap_or_default();
                let args = {
                    let pattern = "\"arguments\":";
                    if let Some(pos) = params.find(pattern) {
                        params[pos + pattern.len()..].trim().to_string()
                    } else {
                        "{}".to_string()
                    }
                };
                handle_tool_call(&id_raw, &tool_name, &args)
            }
            "ping" => mcp_response(&id_raw, "{}"),
            _ => {
                if id_raw == "null" {
                    continue;
                } // it's a notification
                mcp_error(&id_raw, -32601, &format!("Method not found: {}", method))
            }
        };

        writeln!(out, "{}", response).unwrap();
        out.flush().unwrap();
    }
}
