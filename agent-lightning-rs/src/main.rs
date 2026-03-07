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

    // ── Data Layer: Coffee Roasting Profiles ──────────────────────────────────
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
            if coffee_db.profiles.is_empty() {
                println!("  [TIP] Add .csv profiles to: {}", raw_data_dir);
            }
        }
        Err(e) => println!("  [ERROR] Failed to load coffee profiles: {}", e),
    }

    println!("\n✓ All demos completed.\n");
}
