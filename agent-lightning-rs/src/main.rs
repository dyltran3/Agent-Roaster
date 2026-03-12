use agent_roaster::envs::cartpole::CartPole;
use agent_roaster::envs::coffee_roaster::CoffeeRoasterEnv;
use agent_roaster::envs::gridworld::GridWorld;
use agent_roaster::training::coffee_dataset::CoffeeDataset;
/// Agent Lightning RS — Entry Point & Demo
///
/// Demonstrates all three algorithms:
///   1. PPO on GridWorld
///   2. GRPO on CartPole
///   3. LightningRL (Hierarchical) on GridWorld
///
/// Run with: cargo run --release
use agent_roaster::training::config::TrainingConfig;
use agent_roaster::training::training_loop::{train_grpo, train_hierarchical, train_ppo};

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║   ⚡  Agent Lightning RS  —  Rust RL Framework               ║");
    println!("║      Inspired by Microsoft Agent Lightning                    ║");
    println!("║      Core: Tensor Engine + NN + PPO/GRPO/HRL from scratch    ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");

    // ── Demo 1: PINN Agent (PPO) on CoffeeRoasterEnv ─────────────────────────
    println!("\n\n━━━  Demo 1: Physics-Informed Agent on Coffee Roaster  ━━━━━━━━━━━━\n");
    {
        let cfg = TrainingConfig {
            algorithm: "ppo".to_string(),
            environment: "coffee_roaster".to_string(),
            total_episodes: 250,
            log_every: 25,
            eval_every: 50,
            eval_episodes: 2,
            n_steps: 60,
            batch_size: 16,
            update_epochs: 4,
            ..TrainingConfig::default()
        };
        // Run purely inside Physics-Simulation Box
        let mut env = CoffeeRoasterEnv::new();
        let mut eval_env = CoffeeRoasterEnv::new();
        // Return a trained agent for saving
        let trained_agent = train_ppo(&mut env, &mut eval_env, &cfg);

        // Model Persistence
        let path = "roaster_pinn_model.bin";
        match trained_agent.actor.save(path) {
            Ok(_) => println!("  [SUCCESS] Checkpoint saved strictly to {}.", path),
            Err(e) => println!("  [ERROR] Failed to save Checkpoint: {}", e),
        }
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
    let processed_data_dir = "data/roasting_profiles/processed";
    let mut coffee_db = CoffeeDataset::new();
    let _ = coffee_db.load_from_dir(raw_data_dir);
    let _ = coffee_db.load_from_dir(processed_data_dir);

    if !coffee_db.profiles.is_empty() {
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

            use agent_roaster::core::activation::Activation;
            use agent_roaster::core::optimizer::{Adam, Optimizer};
            use agent_roaster::nn::network::Sequential;

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

                    // Extract physical context (assuming mock parsing from dataset)
                    let et = transition.input.data.get(0).copied().unwrap_or(200.0);
                    let bt = transition.input.data.get(1).copied().unwrap_or(150.0);
                    let target_pred = pred.data[0];
                    let target_ror = target_pred - bt; // abstract projection

                    // Compute Loss (Physics-Informed Edge AI)
                    let (loss, grad) = agent_roaster::core::loss::physics_informed_loss(
                        &pred,
                        &transition.output,
                        et,
                        bt,
                        target_ror,
                        0.05,
                    );
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
    }

    // ── Demo 4: End-To-End PINN Inference Workflow (Gray-box AI) ────────────────
    println!("\n\n━━━  Demo 4: End-To-End PINN Inference Workflow (Gray-box AI)  ━━━━\n");
    {
        use agent_roaster::core::activation::Activation;
        use agent_roaster::core::tensor::Tensor;
        use agent_roaster::envs::state_estimator::ExtendedKalmanFilter;
        use agent_roaster::nn::network::Sequential;
        use agent_roaster::security::bounds::{
            apply_hybrid_control, check_safety_bounds, compute_base_gas,
        };

        // 1. Init Physics Math filter and AI Core
        let mut ekf = ExtendedKalmanFilter::new(25.0);
        let policy = Sequential::new()
            .dense(4, 16, Activation::ReLU)
            .dense(16, 16, Activation::ReLU)
            .dense(16, 1, Activation::Tanh); // Output: residual correction in [-1, 1]

        let dt = 1.0;
        let et = 210.0;
        let sensor_bt = 85.0; // Assume Noisy Sensor Data

        println!(
            "  [Step 1] Sensor Readings: ET={:.1}°C, Mũi dò BT={:.1}°C",
            et, sensor_bt
        );

        // 2. State Estimator Filter
        ekf.predict(dt, et);
        ekf.update(sensor_bt);
        let abstract_state = vec![ekf.x[0], ekf.x[1], ekf.x[2], ekf.x[3]];
        println!(
            "  [Step 2] EKF Abstract Vector: T_bean={:.2}, ROR={:.2}, Moisture={:.2}, CDI={:.4}",
            ekf.x[0], ekf.x[1], ekf.x[2], ekf.x[3]
        );

        // 3. Foundation Model Inference
        let state_t = Tensor::new(abstract_state, vec![1, 4]);
        let residual_correction = policy.forward(&state_t).data[0];
        println!(
            "  [Step 3] LLM AI Residual Output: {:.2}%",
            residual_correction * 5.0
        );

        // 4. Hybrid Control Loop
        let target_ror = 15.0; // Expert planned ROR
        let base_gas = compute_base_gas(et, ekf.x[0], target_ror);
        let output_gas = apply_hybrid_control(base_gas, residual_correction);
        println!("  [Step 4] Thermodynamic Base Gas: {:.2}%", base_gas);
        println!("  [Step 5] Final Hybrid Output Gas: {:.2}%", output_gas);

        // 5. Safety Bounds Checker
        let is_safe = check_safety_bounds(et, ekf.x[0], ekf.x[1]);
        if is_safe {
            println!("  [Safety Guard] Status: SAFE ✅ (System operating correctly)");
        } else {
            println!("  [Safety Guard] Status: CRITICAL EMERGENCY STOP ❌ (System limit reached)");
        }
    }

    println!("\n✓ All demos completed.\n");
}
