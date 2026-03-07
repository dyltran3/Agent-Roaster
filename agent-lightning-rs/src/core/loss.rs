/// Loss functions — all return (loss_value: f64, grad_wrt_output: Tensor)
/// to be used directly in backpropagation.
use crate::core::tensor::Tensor;

// ─── Mean Squared Error ───────────────────────────────────────────────────────

/// MSE Loss: L = mean((pred - target)^2)
/// Gradient: dL/d_pred = 2*(pred - target) / n
pub fn mse_loss(pred: &Tensor, target: &Tensor) -> (f64, Tensor) {
    assert_eq!(pred.shape, target.shape, "MSE: pred/target shape mismatch");
    let n = pred.data.len() as f64;
    let diff = pred.sub(target);
    let loss = diff.pow(2.0).sum() / n;
    let grad = diff.mul_scalar(2.0 / n);
    (loss, grad)
}

// ─── Huber Loss (Smooth L1) ───────────────────────────────────────────────────

/// Huber Loss: robust to outliers, transitions between L1 and L2
/// δ = delta (threshold)
pub fn huber_loss(pred: &Tensor, target: &Tensor, delta: f64) -> (f64, Tensor) {
    assert_eq!(pred.shape, target.shape, "Huber: shape mismatch");
    let n = pred.data.len() as f64;
    let mut loss_sum = 0.0;
    let mut grad_data = Vec::with_capacity(pred.data.len());

    for (&p, &t) in pred.data.iter().zip(&target.data) {
        let e = p - t;
        let abs_e = e.abs();
        if abs_e <= delta {
            loss_sum += 0.5 * e * e;
            grad_data.push(e / n);
        } else {
            loss_sum += delta * (abs_e - 0.5 * delta);
            grad_data.push(delta * e.signum() / n);
        }
    }
    (loss_sum / n, Tensor::new(grad_data, pred.shape.clone()))
}

// ─── Cross-Entropy Loss ───────────────────────────────────────────────────────

/// Categorical cross-entropy with probabilities (softmax output)
/// L = -sum(target * log(pred + eps))
pub fn cross_entropy_loss(pred: &Tensor, target: &Tensor) -> (f64, Tensor) {
    assert_eq!(pred.shape, target.shape, "CrossEntropy: shape mismatch");
    let eps = 1e-10;
    let n = pred.data.len() as f64;
    let loss = -pred
        .data
        .iter()
        .zip(&target.data)
        .map(|(&p, &t)| t * (p + eps).ln())
        .sum::<f64>()
        / n;

    // Gradient: -target/pred (combined with softmax grad gives pred-target)
    let grad_data: Vec<f64> = pred
        .data
        .iter()
        .zip(&target.data)
        .map(|(&p, &t)| -t / (p + eps) / n)
        .collect();
    (loss, Tensor::new(grad_data, pred.shape.clone()))
}

/// One-hot cross-entropy: target is a class index, pred is softmax probabilities
/// Returns gradient = pred - one_hot(target) (combined with softmax)
pub fn sparse_cross_entropy_loss(pred: &Tensor, target_idx: usize) -> (f64, Tensor) {
    let eps = 1e-10;
    let loss = -(pred.data[target_idx] + eps).ln();

    // Combined softmax + cross-entropy gradient
    let mut grad = pred.data.clone();
    grad[target_idx] -= 1.0;
    (loss, Tensor::new(grad, pred.shape.clone()))
}

// ─── Policy Gradient / RL-specific losses ────────────────────────────────────

/// PPO Clipped Surrogate Loss
/// L = -mean(min(r*A, clip(r, 1-eps, 1+eps)*A))
/// where r = new_prob / old_prob, A = advantage
pub fn ppo_clip_loss(
    new_log_probs: &Tensor,
    old_log_probs: &Tensor,
    advantages: &Tensor,
    clip_eps: f64,
) -> (f64, Tensor) {
    assert_eq!(new_log_probs.shape, old_log_probs.shape);
    assert_eq!(new_log_probs.shape, advantages.shape);

    let n = new_log_probs.data.len() as f64;
    let mut loss_sum = 0.0;
    let mut grad_data = vec![0.0f64; new_log_probs.data.len()];

    for (i, grad) in grad_data
        .iter_mut()
        .enumerate()
        .take(new_log_probs.data.len())
    {
        let ratio = (new_log_probs.data[i] - old_log_probs.data[i]).exp();
        let adv = advantages.data[i];
        let clip_ratio = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps);
        let obj_unclip = ratio * adv;
        let obj_clip = clip_ratio * adv;
        let obj = obj_unclip.min(obj_clip);
        loss_sum -= obj; // Negate because we minimize (gradient descent)

        // Gradient of unclipped objective
        if obj_unclip <= obj_clip {
            *grad = -adv * ratio / n; // gradient of ratio w.r.t. new_log_prob
        } else {
            // Clipped — zero gradient if outside clip range
            if ratio >= 1.0 - clip_eps && ratio <= 1.0 + clip_eps {
                *grad = -adv * ratio / n;
            }
            // else: gradient is 0 (clipped)
        }
    }

    (
        loss_sum / n,
        Tensor::new(grad_data, new_log_probs.shape.clone()),
    )
}

/// Token-level REINFORCE loss: L = -mean(log_prob * advantage)
/// Gradient: dL/d_log_prob = -advantage / n
pub fn reinforce_loss(log_probs: &Tensor, advantages: &Tensor) -> (f64, Tensor) {
    assert_eq!(log_probs.shape, advantages.shape);
    let n = log_probs.data.len() as f64;
    let loss = -log_probs
        .data
        .iter()
        .zip(&advantages.data)
        .map(|(&lp, &adv)| lp * adv)
        .sum::<f64>()
        / n;

    let grad_data: Vec<f64> = advantages.data.iter().map(|&adv| -adv / n).collect();

    (loss, Tensor::new(grad_data, log_probs.shape.clone()))
}

/// Token-level GRPO loss (simplified)
/// Similar to PPO clip but specifically for group-relative updates
pub fn grpo_loss(
    new_log_probs: &Tensor,
    old_log_probs: &Tensor,
    advantages: &Tensor, // This must be group-relative advantages
    clip_eps: f64,
) -> (f64, Tensor) {
    // GRPO uses the same surrogate objective as PPO but without a critic
    ppo_clip_loss(new_log_probs, old_log_probs, advantages, clip_eps)
}

/// Value function MSE loss for critic
pub fn value_loss(pred_values: &Tensor, returns: &Tensor) -> (f64, Tensor) {
    mse_loss(pred_values, returns)
}

/// Entropy bonus to encourage exploration: H = -sum(p * log(p))
pub fn entropy_loss(probs: &Tensor) -> f64 {
    let eps = 1e-10;
    -probs.data.iter().map(|&p| p * (p + eps).ln()).sum::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_zero() {
        let pred = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let target = pred.clone();
        let (loss, grad) = mse_loss(&pred, &target);
        assert!(loss.abs() < 1e-10);
        assert!(grad.data.iter().all(|&g| g.abs() < 1e-10));
    }

    #[test]
    fn test_huber_small_error() {
        let pred = Tensor::new(vec![1.0], vec![1]);
        let target = Tensor::new(vec![1.1], vec![1]);
        let (loss, _) = huber_loss(&pred, &target, 1.0);
        assert!((loss - 0.005).abs() < 1e-6); // 0.5 * 0.1^2 = 0.005
    }

    #[test]
    fn test_sparse_cross_entropy() {
        // Perfect prediction
        let pred = Tensor::new(vec![0.0, 0.0, 1.0], vec![3]);
        let (loss, _) = sparse_cross_entropy_loss(&pred, 2);
        assert!(loss < 1e-6);
    }
}
