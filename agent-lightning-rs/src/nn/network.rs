use crate::core::activation::Activation;
/// Sequential Neural Network — stacks LinearLayers into a deep network.
///
/// Supports:
///   - forward() → output tensor
///   - forward_with_cache() → (output, layer caches) for backprop
///   - parameters() → all trainable tensors
use crate::core::tensor::Tensor;
use crate::nn::layer::{LayerCache, LinearLayer};

pub struct Sequential {
    pub layers: Vec<LinearLayer>,
}

impl Sequential {
    pub fn new() -> Self {
        Sequential { layers: Vec::new() }
    }

    /// Builder pattern: add a layer
    pub fn add_layer(mut self, layer: LinearLayer) -> Self {
        self.layers.push(layer);
        self
    }

    /// Convenience: add dense layer directly
    pub fn dense(self, in_features: usize, out_features: usize, activation: Activation) -> Self {
        self.add_layer(LinearLayer::new(in_features, out_features, activation))
    }

    /// Forward pass — propagate input through all layers
    pub fn forward(&self, input: &Tensor) -> Tensor {
        // Ensure 2D input (add batch dim if needed)
        let mut current = if input.shape.len() == 1 {
            input.reshape(vec![1, input.data.len()])
        } else {
            input.clone()
        };

        for layer in &self.layers {
            let (output, _) = layer.forward(&current);
            current = output;
        }
        current
    }

    /// Sync weight caches for all layers
    pub fn sync_caches(&mut self) {
        for layer in &mut self.layers {
            layer.sync_cache();
        }
    }

    /// Forward pass with full cache (required for backpropagation)
    pub fn forward_with_cache(&self, input: &Tensor) -> (Tensor, Vec<LayerCache>) {
        let mut caches = Vec::with_capacity(self.layers.len());
        let mut current = if input.shape.len() == 1 {
            input.reshape(vec![1, input.data.len()])
        } else {
            input.clone()
        };

        for layer in &self.layers {
            let (output, cache) = layer.forward(&current);
            caches.push(cache);
            current = output;
        }
        (current, caches)
    }

    /// Backward pass — backpropagate gradients from output to input
    /// Returns gradient w.r.t. the network input
    pub fn backward(&mut self, loss_grad: &Tensor, caches: &[LayerCache]) -> Tensor {
        let mut d_out = loss_grad.clone();
        for (layer, cache) in self.layers.iter_mut().zip(caches.iter()).rev() {
            d_out = layer.backward(&d_out, cache);
        }
        d_out
    }

    /// Zero all parameter gradients
    pub fn zero_grad(&mut self) {
        for layer in &mut self.layers {
            layer.zero_grad();
        }
    }

    /// Total number of trainable parameters
    pub fn num_params(&self) -> usize {
        self.layers.iter().map(|l| l.num_params()).sum()
    }

    /// Get all trainable tensors (for optimizer.step)
    pub fn collect_params(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        for layer in &mut self.layers {
            // Collect weights and biases as separate entries
            params.push(&mut layer.weights as *mut Tensor);
            params.push(&mut layer.bias as *mut Tensor);
        }
        // SAFETY: each pointer is unique and valid for the lifetime of self
        params.into_iter().map(|p| unsafe { &mut *p }).collect()
    }

    pub fn print_summary(&self) {
        println!("═══════════════════════════════════════");
        println!("  Network Architecture");
        println!("───────────────────────────────────────");
        let mut total = 0;
        for (i, layer) in self.layers.iter().enumerate() {
            println!(
                "  Layer {}: {:?}  ({} params)",
                i,
                layer,
                layer.num_params()
            );
            total += layer.num_params();
        }
        println!("───────────────────────────────────────");
        println!("  Total Parameters: {}", total);
        println!("═══════════════════════════════════════");
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Sequential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Sequential({} layers, {} params)",
            self.layers.len(),
            self.num_params()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_forward() {
        let net = Sequential::new()
            .dense(4, 8, Activation::ReLU)
            .dense(8, 2, Activation::Softmax);

        let input = Tensor::zeros(vec![3, 4]); // batch=3
        let output = net.forward(&input);
        assert_eq!(output.shape, vec![3, 2]);
        // Softmax outputs should sum to 1 per row
        let (rows, cols) = output.shape2d();
        for r in 0..rows {
            let row_sum: f64 = (0..cols).map(|c| output.data[r * cols + c]).sum();
            assert!((row_sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_sequential_backward() {
        let mut net =
            Sequential::new()
                .dense(3, 4, Activation::ReLU)
                .dense(4, 2, Activation::Linear);

        let input = Tensor::ones(vec![1, 3]);
        let (output, caches) = net.forward_with_cache(&input);
        let d_loss = Tensor::ones(output.shape.clone());
        let d_input = net.backward(&d_loss, &caches);
        assert_eq!(d_input.shape, vec![1, 3]);
    }
}
