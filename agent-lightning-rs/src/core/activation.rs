/// Activation functions — all operate on Tensor data in-place or return new Tensor.
/// Forward + derivative for backprop.
use crate::core::tensor::Tensor;

// ─── Forward passes ───────────────────────────────────────────────────────────

/// Rectified Linear Unit: f(x) = max(0, x)
pub fn relu(t: &Tensor) -> Tensor {
    Tensor::new(
        t.data.iter().map(|&x| x.max(0.0)).collect(),
        t.shape.clone(),
    )
}

/// Leaky ReLU: f(x) = x if x>0, else alpha*x
pub fn leaky_relu(t: &Tensor, alpha: f64) -> Tensor {
    Tensor::new(
        t.data
            .iter()
            .map(|&x| if x > 0.0 { x } else { alpha * x })
            .collect(),
        t.shape.clone(),
    )
}

/// Sigmoid: f(x) = 1 / (1 + e^(-x))
pub fn sigmoid(t: &Tensor) -> Tensor {
    Tensor::new(
        t.data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect(),
        t.shape.clone(),
    )
}

pub struct GELU;
impl GELU {
    /// Gaussian Error Linear Unit (GELU) approximation
    pub fn forward(inputs: &Tensor) -> Tensor {
        let mut out = Tensor::zeros(inputs.shape.clone());
        for i in 0..inputs.data.len() {
            let x = inputs.data[i];
            out.data[i] =
                0.5 * x * (1.0 + (x * 1.5957691216057308 * (1.0 + 0.044715 * x * x)).tanh());
        }
        out
    }

    pub fn backward(inputs: &Tensor, grad_output: &Tensor) -> Tensor {
        let mut grad_input = Tensor::zeros(inputs.shape.clone());
        for i in 0..inputs.data.len() {
            let x = inputs.data[i];
            let x_sq = x * x;
            let x_cube = x_sq * x;
            let inner = 1.5957691216057308 * (x + 0.044715 * x_cube);
            let tanh_inner = inner.tanh();

            let sech_sq = 1.0 - tanh_inner * tanh_inner;
            let inner_deriv = 1.5957691216057308 * (1.0 + 3.0 * 0.044715 * x_sq);

            let gelu_deriv = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech_sq * inner_deriv;
            grad_input.data[i] = grad_output.data[i] * gelu_deriv;
        }
        grad_input
    }
}

/// Hyperbolic tangent: f(x) = tanh(x)
pub fn tanh_fn(t: &Tensor) -> Tensor {
    Tensor::new(t.data.iter().map(|&x| x.tanh()).collect(), t.shape.clone())
}

/// Softmax over last dimension (1D or row-wise for 2D)
/// Numerically stable: subtract max before exp
pub fn softmax(t: &Tensor) -> Tensor {
    if t.shape.len() == 1 {
        let max_val = t.max();
        let exp_data: Vec<f64> = t.data.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f64 = exp_data.iter().sum();
        Tensor::new(exp_data.iter().map(|x| x / sum).collect(), t.shape.clone())
    } else {
        // Apply row-wise for 2D tensors
        let (rows, cols) = t.shape2d();
        let mut result = vec![0.0f64; rows * cols];
        for r in 0..rows {
            let row_offset = r * cols;
            let row = &t.data[row_offset..row_offset + cols];

            // 1. Find max for numerical stability
            let mut max_val = f64::NEG_INFINITY;
            for &x in row {
                if x > max_val {
                    max_val = x;
                }
            }

            // 2. Calculate sum of exps
            let mut sum = 0.0;
            for (c, &x) in row.iter().enumerate() {
                let e = (x - max_val).exp();
                result[row_offset + c] = e;
                sum += e;
            }

            // 3. Normalize
            let inv_sum = 1.0 / sum;
            for c in 0..cols {
                result[row_offset + c] *= inv_sum;
            }
        }
        Tensor::new(result, t.shape.clone())
    }
}

/// GELU approximation: x * Φ(x), used in transformers
pub fn gelu(t: &Tensor) -> Tensor {
    use std::f64::consts::PI;
    Tensor::new(
        t.data
            .iter()
            .map(|&x| 0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh()))
            .collect(),
        t.shape.clone(),
    )
}

// ─── Derivatives (for backpropagation) ───────────────────────────────────────

/// d/dx ReLU = 1 if x > 0, else 0
pub fn relu_grad(pre_activation: &Tensor) -> Tensor {
    Tensor::new(
        pre_activation
            .data
            .iter()
            .map(|&x| if x > 0.0 { 1.0 } else { 0.0 })
            .collect(),
        pre_activation.shape.clone(),
    )
}

/// d/dx Leaky ReLU
pub fn leaky_relu_grad(pre_activation: &Tensor, alpha: f64) -> Tensor {
    Tensor::new(
        pre_activation
            .data
            .iter()
            .map(|&x| if x > 0.0 { 1.0 } else { alpha })
            .collect(),
        pre_activation.shape.clone(),
    )
}

/// d/dx Sigmoid = sigma(x) * (1 - sigma(x))
pub fn sigmoid_grad(activated: &Tensor) -> Tensor {
    Tensor::new(
        activated.data.iter().map(|&s| s * (1.0 - s)).collect(),
        activated.shape.clone(),
    )
}

/// d/dx Tanh = 1 - tanh(x)^2
pub fn tanh_grad(activated: &Tensor) -> Tensor {
    Tensor::new(
        activated.data.iter().map(|&t| 1.0 - t * t).collect(),
        activated.shape.clone(),
    )
}

/// Softmax Jacobian × upstream gradient (simplified for output layer with cross-entropy)
/// Returns element-wise gradient: softmax_out - one_hot_target
pub fn softmax_cross_entropy_grad(softmax_out: &Tensor, target_idx: usize) -> Tensor {
    let mut grad = softmax_out.data.clone();
    grad[target_idx] -= 1.0;
    Tensor::new(grad, softmax_out.shape.clone())
}

// ─── Activation enum for dynamic dispatch ────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    ReLU,
    LeakyReLU(f64),
    Sigmoid,
    Tanh,
    Softmax,
    GELU,
    Linear, // No activation (identity)
}

impl Activation {
    pub fn forward(&self, t: &Tensor) -> Tensor {
        match self {
            Activation::ReLU => relu(t),
            Activation::LeakyReLU(alpha) => leaky_relu(t, *alpha),
            Activation::Sigmoid => sigmoid(t),
            Activation::Tanh => tanh_fn(t),
            Activation::Softmax => softmax(t),
            Activation::GELU => gelu(t),
            Activation::Linear => t.clone(),
        }
    }

    pub fn backward(&self, cache: &Tensor) -> Tensor {
        match self {
            Activation::ReLU => relu_grad(cache),
            Activation::LeakyReLU(alpha) => leaky_relu_grad(cache, *alpha),
            Activation::Sigmoid => sigmoid_grad(cache),
            Activation::Tanh => tanh_grad(cache),
            Activation::Linear => Tensor::ones(cache.shape.clone()),
            // Softmax backward is handled separately with cross-entropy
            Activation::Softmax | Activation::GELU => Tensor::ones(cache.shape.clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let t = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
        let r = relu(&t);
        assert_eq!(r.data, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sigmoid_range() {
        let t = Tensor::new(vec![-100.0, 0.0, 100.0], vec![3]);
        let r = sigmoid(&t);
        assert!(r.data[0] > 0.0 && r.data[0] < 0.01);
        assert!((r.data[1] - 0.5).abs() < 1e-6);
        assert!(r.data[2] > 0.99 && r.data[2] <= 1.0);
    }

    #[test]
    fn test_softmax_sum_to_one() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let s = softmax(&t);
        let sum: f64 = s.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
}
