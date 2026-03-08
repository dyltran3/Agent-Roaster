use crate::lightning::client::{create_client_server_channels, LightningClient};
use crate::lightning::server::LightningServer;
use crate::rl::env::Environment;
use crate::rl::grpo::{GRPOAgent, GRPOConfig};
use crate::rl::hierarchical::{LightningRLAgent, LightningRLConfig};
use crate::rl::lightning_rl::{
    CreditMode, LightningRL, LightningRLConfig as LRLConfig, TrainerType,
};
use crate::rl::ppo::{PPOAgent, PPOConfig};
use crate::training::config::TrainingConfig;
use crate::training::logger::Logger;
use crate::ui::dashboard::Dashboard;

// ─── PPO Training Loop ────────────────────────────────────────────────────────

pub fn train_ppo(
    env: &mut dyn Environment,
    eval_env: &mut dyn Environment,
    cfg: &TrainingConfig,
) -> PPOAgent {
    let ppo_cfg = PPOConfig {
        lr_actor: cfg.lr_actor,
        lr_critic: cfg.lr_critic,
        gamma: cfg.gamma,
        lambda: cfg.lambda,
        clip_eps: cfg.clip_eps,
        entropy_coef: cfg.entropy_coef,
        value_coef: 0.5,
        update_epochs: cfg.update_epochs,
        batch_size: cfg.batch_size,
        n_steps: cfg.n_steps,
    };

    let state_dim = env.state_space();
    let action_dim = env.action_space();

    // 1. Setup Server & Orchestrator
    let ppo_agent = PPOAgent::new(state_dim, action_dim, ppo_cfg);
    let lrl_cfg = LRLConfig {
        credit_mode: CreditMode::GAE,
        gamma: cfg.gamma,
    };
    let mut server = LightningServer::new(1); // Update every 1 episode for demo
    server.trainer = Some(LightningRL::new(TrainerType::PPO(ppo_agent), lrl_cfg));

    // 2. Setup Client
    let channels = create_client_server_channels();
    server.register_client(channels.server_tx);
    let mut client = LightningClient::new(0, channels.client_tx, channels.server_rx);

    let logger = Logger::new();
    let mut dashboard = Dashboard::new();
    dashboard.status = format!("Training on {}", env.name());

    logger.print_header("Agent Lightning PPO (Disaggregated)", env.name());

    for episode in 0..cfg.total_episodes {
        // --- CLIENT SIDE: Execution ---
        let mut state = env.reset();
        let mut total_reward = 0.0;
        let mut done = false;

        while !done {
            // In a real disaggregated setup, the client calls the policy
            // Here we use the server's trainer for local simulation
            let actor = match &server.trainer {
                Some(LightningRL {
                    trainer: TrainerType::PPO(a),
                    ..
                }) => a,
                _ => panic!("No PPO trainer"),
            };

            let (action, log_prob, _value_est) = actor.select_action(&state);
            let result = env.step(action);
            total_reward += result.reward;

            // Trace to server: Unified MDP Transition
            let input = crate::core::tensor::Tensor::new(state.clone(), vec![1, state.len()]);
            let output = crate::core::tensor::Tensor::new(vec![action as f64], vec![1]);
            let transition = crate::rl::transition::Transition::new(
                input,
                output,
                vec![log_prob],
                vec![result.reward],
            );
            client.trace_transition(transition);

            state = result.next_state;
            done = result.done;
        }
        client.episode_done(total_reward, Vec::new());

        // --- SERVER SIDE: Training ---
        server.process_messages(&channels.client_rx);

        // Update UI
        dashboard.update(total_reward, 0.0, server.stats.policy_version as u32);
        // Render every episode for real-time feel
        dashboard.render();

        if episode % cfg.log_every == 0 {
            // logger.log_episode(episode, total_reward, 0.0, 0.0);
        }

        // Evaluation run
        if episode % cfg.eval_every == 0 {
            // Simplified eval using server's agent
            let agent = match &server.trainer {
                Some(LightningRL {
                    trainer: TrainerType::PPO(a),
                    ..
                }) => a,
                _ => panic!("No PPO trainer"),
            };
            let eval_reward = evaluate(agent, eval_env, cfg.eval_episodes);
            logger.log_eval(episode, eval_reward);
        }
    }

    logger.print_footer();

    // Extract Agent (Destructure TrainerType)
    match server.trainer.unwrap().trainer {
        TrainerType::PPO(agent) => agent,
        _ => panic!("Expected PPO agent"),
    }
}

// ─── GRPO Training Loop ────────────────────────────────────────────────────────

pub fn train_grpo(env: &mut dyn Environment, cfg: &TrainingConfig) {
    let grpo_cfg = GRPOConfig {
        lr: cfg.lr_actor,
        gamma: cfg.gamma,
        group_size: cfg.group_size,
        clip_eps: cfg.clip_eps,
        kl_coef: cfg.kl_coef,
        n_episodes_per_update: cfg.group_size,
        max_steps: cfg.n_steps,
    };

    let state_dim = env.state_space();
    let action_dim = env.action_space();

    // Setup orchestrated trainer
    let grpo_agent = GRPOAgent::new(state_dim, action_dim, grpo_cfg);
    let lrl_cfg = LRLConfig {
        credit_mode: CreditMode::Uniform,
        gamma: cfg.gamma,
    };
    let mut server = LightningServer::new(cfg.group_size); // Update per group
    server.trainer = Some(LightningRL::new(TrainerType::GRPO(grpo_agent), lrl_cfg));

    let mut logger = Logger::new();
    logger.print_header("Agent Lightning GRPO (Disaggregated)", env.name());

    for episode in 0..cfg.total_episodes {
        // Collect one episode (local for simulation)
        let mut state = env.reset();
        let mut total_reward = 0.0;
        let mut transitions = Vec::new();

        for _ in 0..cfg.n_steps {
            let agent = match &server.trainer {
                Some(LightningRL {
                    trainer: TrainerType::GRPO(a),
                    ..
                }) => a,
                _ => panic!("No GRPO trainer"),
            };
            let action = agent.select_action(&state);
            let state_t = crate::core::tensor::Tensor::new(state.clone(), vec![1, state.len()]);
            let probs = agent.policy.forward(&state_t);
            let log_prob = (probs.data[action] + 1e-10).ln();

            let result = env.step(action);
            total_reward += result.reward;

            transitions.push(crate::rl::transition::Transition::new(
                state_t,
                crate::core::tensor::Tensor::new(vec![action as f64], vec![1]),
                vec![log_prob],
                vec![result.reward],
            ));

            if result.done {
                break;
            }
            state = result.next_state;
        }

        // In GRPO, the server handles multiple episodes before triggering update
        // Here we just manually feed them
        for t in transitions {
            server.handle_message(crate::lightning::client::ClientToServerMsg::Transition(t));
        }
        server.handle_message(crate::lightning::client::ClientToServerMsg::EpisodeDone {
            episode_id: episode as u64,
            total_reward,
            intermediate_rewards: Vec::new(),
        });

        server.trigger_training_update();

        if episode % cfg.log_every == 0 {
            logger.log_episode(episode, total_reward, 0.0, 0.0);
        }
    }

    logger.print_footer();
}

// ─── Hierarchical RL Training Loop ────────────────────────────────────────────

pub fn train_hierarchical(env: &mut dyn Environment, cfg: &TrainingConfig) {
    // Keeping this legacy implementation for now as it's already complex
    let h_cfg = LightningRLConfig {
        lr_manager: cfg.lr_manager,
        lr_worker: cfg.lr_worker,
        gamma_manager: cfg.gamma,
        gamma_worker: cfg.gamma,
        n_subgoals: cfg.n_subgoals,
        subgoal_horizon: cfg.subgoal_horizon,
        intrinsic_reward_scale: 1.0,
        max_steps_per_episode: cfg.n_steps,
    };

    let state_dim = env.state_space();
    let action_dim = env.action_space();
    let mut agent = LightningRLAgent::new(state_dim, action_dim, h_cfg);
    let mut logger = Logger::new();

    logger.print_header("Hierarchical RL", env.name());

    for episode in 0..cfg.total_episodes {
        let reward = agent.run_episode(env);

        if episode % cfg.log_every == 0 {
            logger.log_episode(episode, reward, 0.0, 0.0);
        }
    }

    logger.print_footer();
}

// ─── Evaluation Helper ────────────────────────────────────────────────────────

fn evaluate(agent: &PPOAgent, env: &mut dyn Environment, n_episodes: usize) -> f64 {
    use crate::core::tensor::Tensor;
    use crate::rl::ppo::greedy_action;

    let mut total_reward = 0.0;
    for _ in 0..n_episodes {
        let mut state = env.reset();
        let mut done = false;
        while !done {
            let state_t = Tensor::new(state.clone(), vec![1, state.len()]);
            let probs = agent.actor.forward(&state_t);
            let action = greedy_action(&probs.data);
            let result = env.step(action);
            total_reward += result.reward;
            done = result.done;
            state = result.next_state;
        }
    }
    total_reward / n_episodes as f64
}
