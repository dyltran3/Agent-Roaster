//! Backpropagation trainer — orchestrates forward + backward + optimizer step.
//! Wraps a Sequential network + Optimizer into a single training unit.

use crate::core::optimizer::Optimizer;
use crate::core::tensor::Tensor;
use crate::nn::network::Sequential;

pub struct Backprop<O: Optimizer> {
    pub network: Sequential,
    pub optimizer: O,
}

impl<O: Optimizer> Backprop<O> {
    pub fn new(network: Sequential, optimizer: O) -> Self {
        Backprop { network, optimizer }
    }

    /// One training step: forward → loss → backward → optimizer step
    /// Returns loss value
    pub fn train_step<F>(&mut self, input: &Tensor, loss_fn: F) -> f64
    where
        F: Fn(&Tensor) -> (f64, Tensor), // (loss_value, grad_wrt_output)
    {
        // 1. Zero gradients
        self.network.zero_grad();

        // 2. Forward pass with cache
        let (output, caches) = self.network.forward_with_cache(input);

        // 3. Compute loss and upstream gradient
        let (loss, loss_grad) = loss_fn(&output);

        // 4. Backward pass
        self.network.backward(&loss_grad, &caches);

        // 5. Optimizer step
        let mut params = self.network.collect_params();
        self.optimizer.step(&mut params);

        loss
    }

    /// Inference only (no gradient compute)
    pub fn predict(&self, input: &Tensor) -> Tensor {
        self.network.forward(input)
    }
}
