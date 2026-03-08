use agent_lightning::envs::cartpole::CartPole;
use agent_lightning::envs::gridworld::GridWorld;
use agent_lightning::training::coffee_dataset::CoffeeDataset;
/// Agent Lightning RS — Entry Point & Demo
///
/// Demonstrates all three algorithms:
///   1. PPO on GridWorld
///   2. GRPO on CartPole
///   3. LightningRL (Hierarchical) on GridWorld
///
/// Run with: cargo run --release
use agent_lightning::training::config::TrainingConfig;
use agent_lightning::training::training_loop::{train_grpo, train_hierarchical, train_ppo};

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║   ⚡  Agent Lightning RS  —  Rust RL Framework               ║");
    println!("║      Inspired by Microsoft Agent Lightning                    ║");
    println!("║      Core: Tensor Engine + NN + PPO/GRPO/HRL from scratch    ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");

    // ── Demo 1: PPO on GridWorld ──────────────────────────────────────────────
    println!("\n\n━━━  Demo 1: PPO Agent on GridWorld  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    {
        let cfg = TrainingConfig {
            algorithm: "ppo".to_string(),
            environment: "gridworld".to_string(),
            total_episodes: 300,
            log_every: 25,
            eval_every: 100,
            eval_episodes: 5,
            n_steps: 128,
            batch_size: 32,
            update_epochs: 4,
            ..TrainingConfig::default()
        };
        let mut env = GridWorld::new(5, 5);
        let mut eval_env = GridWorld::new(5, 5);
        train_ppo(&mut env, &mut eval_env, &cfg);
    }

    // ── Demo 2: GRPO on CartPole ──────────────────────────────────────────────
    println!("\n\n━━━  Demo 2: GRPO Agent on CartPole  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    {
        let cfg = TrainingConfig {
            algorithm: "grpo".to_string(),
            environment: "cartpole".to_string(),
            total_episodes: 200,
            log_every: 20,
            group_size: 4,
            kl_coef: 0.01,
            ..TrainingConfig::default()
        };
        let mut env = CartPole::new();
        train_grpo(&mut env, &cfg);
    }

    // ── Demo 3: LightningRL (Hierarchical) on GridWorld ───────────────────────
    println!("\n\n━━━  Demo 3: LightningRL (Hierarchical) on GridWorld  ━━━━━━━━━━━━\n");
    {
        let cfg = TrainingConfig {
            algorithm: "hierarchical".to_string(),
            environment: "gridworld".to_string(),
            total_episodes: 200,
            log_every: 20,
            n_subgoals: 4,
            subgoal_horizon: 8,
            lr_manager: 1e-4,
            lr_worker: 3e-4,
            n_steps: 100,
            ..TrainingConfig::default()
        };
        let mut env = GridWorld::new(5, 5);
        train_hierarchical(&mut env, &cfg);
    }

    println!("\n\n━━━  Data Layer: Coffee Roasting Profiles Analysis  ━━━━━━━━━━━\n");
    let raw_data_dir = "data/roasting_profiles/raw";
    let mut coffee_db = CoffeeDataset::new();
    match coffee_db.load_from_dir(raw_data_dir) {
        Ok(_) => {
            println!(
                "  [INFO] Discovered {} expert roasting profiles.",
                coffee_db.profiles.len()
            );
            let transitions = coffee_db.to_transitions();
            println!(
                "  [INFO] Extracted {} token-level transitions for training.",
                transitions.len()
            );

            if !transitions.is_empty() {
                println!("\n  [EVALUATION] Starting Offline RL (Behavioral Cloning) with strict constraints...");

                use agent_lightning::core::activation::Activation;
                use agent_lightning::core::loss::mse_loss;
                use agent_lightning::core::optimizer::{Adam, Optimizer};
                use agent_lightning::nn::network::Sequential;

                // 1. Build a Neural Network for prediction (5 state variables -> 1 target_temp)
                let mut policy = Sequential::new()
                    .dense(5, 16, Activation::ReLU)
                    .dense(16, 16, Activation::ReLU)
                    .dense(16, 1, Activation::Linear); // Output: Target Temp

                let mut optimizer = Adam::new(1e-3).with_betas(0.9, 0.999);
                let epochs = 300;

                for epoch in 1..=epochs {
                    let mut total_loss = 0.0;

                    for transition in &transitions {
                        // Forward with cache for backprop
                        let (pred, caches) = policy.forward_with_cache(&transition.input);

                        // Compute Loss
                        let (loss, grad) = mse_loss(&pred, &transition.output); // holds target_temp
                        total_loss += loss;

                        // Backprop via Framework Standard
                        policy.zero_grad();
                        policy.backward(&grad, &caches);

                        // Optimize
                        let mut params = policy.collect_params();
                        optimizer.step(&mut params);
                        policy.sync_caches();
                    }

                    if epoch % 50 == 0 || epoch == 1 {
                        println!(
                            "    -> Epoch {:3} | MSE Loss: {:.4}",
                            epoch,
                            total_loss / transitions.len() as f64
                        );
                    }
                }

                println!("\n  [EVALUATION REPORT]");
                // Test the first transition
                let test_state = &transitions[0].input;
                let test_target = &transitions[0].output.data[0];
                let pred_target = policy.forward(test_state).data[0];

                println!("    * State Input: {:.2?}", test_state.data);
                println!("    * Expert Target Temp : {:.2} °C", test_target);
                println!("    * AI Predicted Temp  : {:.2} °C", pred_target);
                let error = (test_target - pred_target).abs();
                println!("    * Absolute Error     : {:.2} °C", error);

                if error > 5.0 {
                    println!("\n  [VERDICT] ❌ Model struggle to memorize the sequence. The pure linear network might be too weak (underfitting) or lacks non-linear activation like GELU to represent complex thermal dynamics.");
                } else {
                    println!("\n  [VERDICT] ✅ Model successfully cloned the expert profile within acceptable margin. Evaluation Passed!");
                }
            }

            if coffee_db.profiles.is_empty() {
                println!("  [TIP] Add .csv profiles to: {}", raw_data_dir);
            }
        }
        Err(e) => println!("  [ERROR] Failed to load coffee profiles: {}", e),
    }

    println!("\n✓ All demos completed.\n");
}
