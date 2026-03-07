//! Optimizers — SGD and Adam, applied to collections of (param, grad) pairs.
//! All optimizers update params in-place based on accumulated gradients.

use crate::core::tensor::Tensor;

// ─── Optimizer Trait ──────────────────────────────────────────────────────────

pub trait Optimizer: Send {
    /// Apply one optimization step.
    /// params: mutable slice of tensors (weights/biases)
    fn step(&mut self, params: &mut [&mut Tensor]);

    /// Zero out all accumulated gradients
    fn zero_grad(params: &mut [&mut Tensor]) {
        for p in params.iter_mut() {
            p.zero_grad();
        }
    }
}

// ─── SGD ─────────────────────────────────────────────────────────────────────

pub struct SGD {
    pub lr: f64,
    pub momentum: f64,
    /// Momentum velocity buffers, keyed by param index
    velocities: Vec<Vec<f64>>,
}

impl SGD {
    pub fn new(lr: f64, momentum: f64) -> Self {
        SGD {
            lr,
            momentum,
            velocities: Vec::new(),
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, params: &mut [&mut Tensor]) {
        // Lazily initialize velocity buffers
        if self.velocities.len() != params.len() {
            self.velocities = params.iter().map(|p| vec![0.0; p.data.len()]).collect();
        }

        for (i, param) in params.iter_mut().enumerate() {
            if let Some(ref grad) = param.grad.clone() {
                for (j, (&g, w)) in grad.iter().zip(param.data.iter_mut()).enumerate() {
                    self.velocities[i][j] = self.momentum * self.velocities[i][j] - self.lr * g;
                    *w += self.velocities[i][j];
                }
            }
        }
    }
}

// ─── Adam ─────────────────────────────────────────────────────────────────────

/// Adam optimizer with optional weight decay (AdamW variant)
pub struct Adam {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
    /// First moment (mean of gradient)
    m: Vec<Vec<f64>>,
    /// Second moment (uncentered variance)
    v: Vec<Vec<f64>>,
    /// Step counter (for bias correction)
    t: u64,
}

impl Adam {
    pub fn new(lr: f64) -> Self {
        Adam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }

    pub fn with_weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }

    pub fn with_betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }
}

impl Optimizer for Adam {
    fn step(&mut self, params: &mut [&mut Tensor]) {
        // Lazily initialize moment buffers
        if self.m.len() != params.len() {
            self.m = params.iter().map(|p| vec![0.0; p.data.len()]).collect();
            self.v = params.iter().map(|p| vec![0.0; p.data.len()]).collect();
        }

        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        for (i, param) in params.iter_mut().enumerate() {
            if let Some(ref grad) = param.grad.clone() {
                for (j, (&g, w)) in grad.iter().zip(param.data.iter_mut()).enumerate() {
                    // Update moments
                    self.m[i][j] = self.beta1 * self.m[i][j] + (1.0 - self.beta1) * g;
                    self.v[i][j] = self.beta2 * self.v[i][j] + (1.0 - self.beta2) * g * g;

                    // Bias-corrected estimates
                    let m_hat = self.m[i][j] / bc1;
                    let v_hat = self.v[i][j] / bc2;

                    // AdamW: apply weight decay directly to param
                    let wd_term = self.weight_decay * *w;

                    // Update param
                    *w -= self.lr * (m_hat / (v_hat.sqrt() + self.epsilon) + wd_term);
                }
            }
        }
    }
}

// ─── RMSProp ──────────────────────────────────────────────────────────────────

pub struct RMSProp {
    pub lr: f64,
    pub rho: f64,
    pub epsilon: f64,
    cache: Vec<Vec<f64>>,
}

impl RMSProp {
    pub fn new(lr: f64) -> Self {
        RMSProp {
            lr,
            rho: 0.9,
            epsilon: 1e-8,
            cache: Vec::new(),
        }
    }
}

impl Optimizer for RMSProp {
    fn step(&mut self, params: &mut [&mut Tensor]) {
        if self.cache.len() != params.len() {
            self.cache = params.iter().map(|p| vec![0.0; p.data.len()]).collect();
        }

        for (i, param) in params.iter_mut().enumerate() {
            if let Some(ref grad) = param.grad.clone() {
                for (j, (&g, w)) in grad.iter().zip(param.data.iter_mut()).enumerate() {
                    self.cache[i][j] = self.rho * self.cache[i][j] + (1.0 - self.rho) * g * g;
                    *w -= self.lr * g / (self.cache[i][j].sqrt() + self.epsilon);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_param_with_grad(val: f64, grad: f64) -> Tensor {
        let mut t = Tensor::new(vec![val], vec![1]).with_grad();
        t.accumulate_grad(&[grad]);
        t
    }

    #[test]
    fn test_sgd_decreases_param() {
        let mut p = make_param_with_grad(1.0, 1.0);
        let mut optim = SGD::new(0.1, 0.0);
        {
            let mut params: Vec<&mut Tensor> = vec![&mut p];
            optim.step(&mut params);
        }
        assert!(
            p.data[0] < 1.0,
            "SGD should decrease param when grad is positive"
        );
    }

    #[test]
    fn test_adam_decreases_param() {
        let mut p = make_param_with_grad(1.0, 1.0);
        let mut optim = Adam::new(0.01);
        {
            let mut params: Vec<&mut Tensor> = vec![&mut p];
            optim.step(&mut params);
        }
        assert!(
            p.data[0] < 1.0,
            "Adam should decrease param when grad is positive"
        );
    }
}
