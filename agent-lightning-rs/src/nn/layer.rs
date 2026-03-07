use crate::core::activation::Activation;
/// Linear Layer — the fundamental building block of a neural network.
///
/// Forward:  output = input @ weights.T + bias
/// Backward: computes gradients for weights, bias, and input
///
/// Shape conventions:
///   input:   (batch, in_features)
///   weights: (out_features, in_features)
///   bias:    (1, out_features)
///   output:  (batch, out_features)
use crate::core::tensor::Tensor;

// ─── Layer Cache (for backprop) ───────────────────────────────────────────────

#[derive(Clone)]
pub struct LayerCache {
    pub input: Tensor,          // Pre-activation input to layer
    pub pre_activation: Tensor, // Wx + b before activation
    pub output: Tensor,         // After activation
}

// ─── Linear Layer ─────────────────────────────────────────────────────────────

pub struct LinearLayer {
    pub weights: Tensor, // (out_features, in_features)
    pub bias: Tensor,    // (1, out_features)
    pub activation: Activation,
    pub in_features: usize,
    pub out_features: usize,
    cached_weights_t: Option<Tensor>,
}

impl LinearLayer {
    pub fn new(in_features: usize, out_features: usize, activation: Activation) -> Self {
        let weights = Tensor::xavier(out_features, in_features).with_grad();
        let bias = Tensor::zeros(vec![1, out_features]).with_grad();
        LinearLayer {
            weights,
            bias,
            activation,
            in_features,
            out_features,
            cached_weights_t: None,
        }
    }

    /// Synchronize the transposed weights cache. Call this after weight updates.
    pub fn sync_cache(&mut self) {
        self.cached_weights_t = Some(self.weights.transpose());
    }

    /// Forward pass: returns (output, cache_for_backprop)
    /// input: (batch, in_features)
    pub fn forward(&self, input: &Tensor) -> (Tensor, LayerCache) {
        // pre_activation = input @ weights.T + bias
        // Using optimized matmul. If we have a cached transpose, we could optimize further,
        // but matmul(weights.T) will transpose it back to weights.
        // For real-time inference, it's better to avoid matmul overhead entirely.

        let (batch, _) = input.shape2d();
        let out = self.out_features;
        let mut biased_data = vec![0.0f64; batch * out];

        // Optimized Linear Forward: dot product directly with each weight row
        for b in 0..batch {
            let in_row = &input.data[b * self.in_features..(b + 1) * self.in_features];
            for o in 0..out {
                let w_row = &self.weights.data[o * self.in_features..(o + 1) * self.in_features];
                let mut sum = self.bias.data[o];
                for (i, &x) in in_row.iter().enumerate() {
                    sum += x * w_row[i];
                }
                biased_data[b * out + o] = sum;
            }
        }

        let pre_activation = Tensor::new(biased_data, vec![batch, out]);
        let output = self.activation.forward(&pre_activation);

        let cache = LayerCache {
            input: input.clone(),
            pre_activation,
            output: output.clone(),
        };
        (output, cache)
    }

    /// Backward pass: given upstream gradient, compute:
    ///   - d_weights: gradient wrt weights
    ///   - d_bias:    gradient wrt bias
    ///   - d_input:   gradient wrt input (to propagate to previous layer)
    pub fn backward(&mut self, d_out: &Tensor, cache: &LayerCache) -> Tensor {
        // Step 1: backprop through activation
        let act_grad = self.activation.backward(&cache.pre_activation);
        let d_pre_act = d_out.mul(&act_grad); // element-wise: (batch, out_features)

        let (batch, out) = d_pre_act.shape2d();
        let in_f = self.in_features;

        // Step 2: gradient wrt weights = d_pre_act.T @ input  → (out, in_f)
        let d_w = d_pre_act.transpose().matmul(&cache.input);

        // Step 3: gradient wrt bias = sum over batch dimension
        let mut d_bias_data = vec![0.0f64; out];
        for b in 0..batch {
            for (o, bias) in d_bias_data.iter_mut().enumerate().take(out) {
                *bias += d_pre_act.data[b * out + o];
            }
        }
        let d_bias = Tensor::new(d_bias_data, vec![1, out]);

        // Step 4: gradient wrt input = d_pre_act @ weights  → (batch, in_f)
        let d_input = d_pre_act.matmul(&self.weights);

        // Step 5: accumulate gradients
        self.weights.accumulate_grad(&d_w.data);
        self.bias.accumulate_grad(&d_bias.data);

        assert_eq!(d_input.shape, vec![batch, in_f]);
        d_input
    }

    /// Get all trainable parameters (for optimizer)
    pub fn params(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weights, &mut self.bias]
    }

    /// Zero gradient buffers
    pub fn zero_grad(&mut self) {
        self.weights.zero_grad();
        self.bias.zero_grad();
    }

    pub fn num_params(&self) -> usize {
        self.weights.numel() + self.bias.numel()
    }
}

impl std::fmt::Debug for LinearLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Linear({} → {}, {:?})",
            self.in_features, self.out_features, self.activation
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_forward_shape() {
        let layer = LinearLayer::new(4, 3, Activation::ReLU);
        let input = Tensor::ones(vec![2, 4]); // batch=2, features=4
        let (output, _) = layer.forward(&input);
        assert_eq!(output.shape, vec![2, 3]);
    }

    #[test]
    fn test_linear_backward_produces_gradients() {
        let mut layer = LinearLayer::new(3, 2, Activation::Linear);
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let (output, cache) = layer.forward(&input);
        let d_out = Tensor::ones(output.shape.clone());
        let d_input = layer.backward(&d_out, &cache);

        assert_eq!(d_input.shape, vec![2, 3]);
        assert!(layer.weights.grad.is_some());
        assert!(layer.bias.grad.is_some());
    }
}
