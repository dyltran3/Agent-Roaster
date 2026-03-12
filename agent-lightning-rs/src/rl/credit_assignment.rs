/// LightningRL Credit Assignment (Step 1 of the algorithm)
///
/// This module is responsible for distributing the return of an episode
/// back to the individual transitions/actions.
use crate::rl::transition::Transition;

/// Assigns credit to each action in an episode.
///
/// According to the Agent Lightning paper, a simple uniform credit assignment
/// can be very effective: assigning the total episode return to every action.
pub fn assign_uniform_credit(episode: &mut [Transition]) {
    let total_return: f64 = episode.iter().map(|t| t.total_reward()).sum();

    for transition in episode.iter_mut() {
        // Every token in the output of this transition gets the same total_return advantage
        let n_tokens = transition.rewards.len();
        transition.advantages = vec![total_return; n_tokens];
    }
}

/// Assigns credit using discounted future returns (standard RL return calculation).
///
/// advantage_t = sum_{k=t}^{T} gamma^{k-t} * reward_k
pub fn assign_discounted_credit(episode: &mut [Transition], gamma: f64) {
    let mut running_return = 0.0;

    // We iterate backwards through the entire sequence of steps in the episode
    // Note: Since each transition might have multiple internal tokens/rewards,
    // we need to handle that flat or nested. Here we treat each Transition as
    // a single semantic unit for return calculation if needed, or per-token if rewards are per-token.

    // The paper treats (Input, Output, Reward) as the unit.
    // If rewards are per-token, we work on the flattened reward sequence.

    let all_rewards: Vec<f64> = episode
        .iter()
        .flat_map(|t| t.rewards.iter())
        .cloned()
        .collect();
    let n_total = all_rewards.len();
    let mut all_advantages = vec![0.0; n_total];

    for i in (0..n_total).rev() {
        running_return = all_rewards[i] + gamma * running_return;
        all_advantages[i] = running_return;
    }

    // Distribute back to transitions
    let mut offset = 0;
    for transition in episode.iter_mut() {
        let n_tokens = transition.rewards.len();
        transition.advantages = all_advantages[offset..offset + n_tokens].to_vec();
        offset += n_tokens;
    }
}

/// Advantage calculation using GAE (Generalized Advantage Estimation).
/// Requires value estimates from a critic network.
pub fn assign_gae_credit(
    episode: &mut [Transition],
    values: &[f64], // Value estimates for each step/token
    gamma: f64,
    lambda: f64,
) {
    let all_rewards: Vec<f64> = episode
        .iter()
        .flat_map(|t| t.rewards.iter())
        .cloned()
        .collect();
    let n_total = all_rewards.len();
    assert_eq!(
        values.len(),
        n_total,
        "Values must match total number of reward steps"
    );

    let mut all_advantages = vec![0.0; n_total];
    let mut gae = 0.0;

    for i in (0..n_total).rev() {
        let next_value = if i + 1 < n_total { values[i + 1] } else { 0.0 };
        let delta = all_rewards[i] + gamma * next_value - values[i];
        gae = delta + gamma * lambda * gae;
        all_advantages[i] = gae;
    }

    // Distribute back to transitions
    let mut offset = 0;
    for transition in episode.iter_mut() {
        let n_tokens = transition.rewards.len();
        transition.advantages = all_advantages[offset..offset + n_tokens].to_vec();
        offset += n_tokens;
    }
}

/// Normalizes advantages across a batch of transitions for training stability.
pub fn normalize_advantages(transitions: &mut [Transition]) {
    let all_advs: Vec<f64> = transitions
        .iter()
        .flat_map(|t| t.advantages.iter())
        .cloned()
        .collect();
    if all_advs.is_empty() {
        return;
    }

    let n = all_advs.len() as f64;
    let mean = all_advs.iter().sum::<f64>() / n;
    let var = all_advs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std = (var + 1e-8).sqrt();

    for transition in transitions.iter_mut() {
        for adv in transition.advantages.iter_mut() {
            if std < 1e-4 {
                *adv = 0.0; // Prevent explosion if variance is extremely low
            } else {
                *adv = (*adv - mean) / std;
            }
        }
    }
}

/// Physics-Informed Reward Shaping for Coffee Roasting Env (Edge AI)
/// Encodes Maillard phase curve (F-13), Color development (F-15),
/// and precision First Crack Timing Model (F-38, F-40).
pub fn calculate_physics_reward(
    bt: f64,
    ror: f64,
    target_ror: f64,
    is_first_crack: bool,
    crack_timing: f64,
    expected_crack_timing: f64,
) -> f64 {
    let mut reward = 0.0;

    // 1. ROR tracking precision reward
    let ror_error = (ror - target_ror).abs();
    reward += if ror_error < 1.0 {
        1.0
    } else {
        -ror_error * 0.5
    };

    // 2. Maillard Reaction & Color Dev (F-13, F-15)
    // Between 150C and 200C is the Maillard zone where chemical complexity builds.
    if bt > 150.0 && bt < 200.0 {
        // Ideal Maillard ROR should be smoothly decaying, typically 5-15 C/min
        if ror > 5.0 && ror < 15.0 {
            reward += 2.0; // Positive reinforcement for correct Maillard pacing
        } else {
            reward -= 2.0; // Penalty for baking (too slow) or scorching (too fast)
        }
    }

    // 3. First Crack Timing Penalty/Reward (F-38, F-40)
    // Hitting the expected crack window is the ultimate test of roasting mastery.
    if is_first_crack {
        let timing_error = (crack_timing - expected_crack_timing).abs();
        if timing_error < 30.0 {
            reward += 10.0; // Huge reward for hitting precise crack window
        } else {
            reward -= timing_error * 0.2; // Penalty for prematurely or late crack
        }
    }

    reward
}
