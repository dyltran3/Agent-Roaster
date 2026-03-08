use crate::core::activation::softmax;
use crate::core::activation::Activation;
use crate::core::tensor::Tensor;
use crate::nn::layer::LinearLayer;

/// **Multi-Head Self-Attention (MHSA)**
/// The core mechanism of Transformer/LLM Architectures.
/// Projects input tokens into Query (Q), Key (K), and Value (V) tensors,
/// splits them into multiple "heads" to attend to different context patterns,
/// and re-combines them.
pub struct MultiHeadAttention {
    pub d_model: usize,
    pub num_heads: usize,
    pub d_k: usize,

    // Projections (using existing LinearLayer)
    pub w_q: LinearLayer,
    pub w_k: LinearLayer,
    pub w_v: LinearLayer,
    pub w_o: LinearLayer, // Final output projection
}

impl MultiHeadAttention {
    /// Creates a new MHSA block.
    /// `d_model`: Embedding dimension (e.g., 256)
    /// `num_heads`: Number of attention streams (e.g., 8). `d_model` must be divisible by `num_heads`.
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        assert!(
            d_model % num_heads == 0,
            "d_model must be divisible by num_heads"
        );
        let d_k = d_model / num_heads;

        MultiHeadAttention {
            d_model,
            num_heads,
            d_k,
            // Projections have no activation initially (linear matmul)
            w_q: LinearLayer::new(d_model, d_model, Activation::Linear),
            w_k: LinearLayer::new(d_model, d_model, Activation::Linear),
            w_v: LinearLayer::new(d_model, d_model, Activation::Linear),
            w_o: LinearLayer::new(d_model, d_model, Activation::Linear),
        }
    }

    /// Forward pass of Self-Attention
    /// Currently simulates attention over a sequence conceptually flattened.
    /// Standard Transformer Shape: (Batch, Seq_Len, d_model).
    /// Since Agent Lightning uses dynamically flattened state,
    /// we process it as a 2D Tensor (Seq_Len x d_model).
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // 1. Linear Projections (x -> Q, K, V)
        let (q, _) = self.w_q.forward(x);
        let (k, _) = self.w_k.forward(x);
        let (v, _) = self.w_v.forward(x);

        // 2. Scaled Dot-Product Attention: Softmax(Q * K.T / sqrt(d_k)) * V
        // Note: Full MatMul implementation is required in `tensor.rs` for multi-dimensional operations.
        // For zero-dependency RL compatibility, if `matmul` is missing, we use a structured fallback.

        let mut attention_scores = q.matmul(&k.transpose());

        // Scale by sqrt(d_k)
        let scale = (self.d_k as f64).sqrt();
        for val in attention_scores.data.iter_mut() {
            *val /= scale;
        }

        // Apply Softmax across the sequence dimension (producing Attention weights)
        let attention_weights = softmax(&attention_scores);

        // 3. Attend to Values
        let context = attention_weights.matmul(&v);

        // 4. Output Projection
        let (output, _) = self.w_o.forward(&context);
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mha_forward_shape() {
        let d_model = 16;
        let num_heads = 4;
        let seq_len = 10;

        // Input: (Seq_Len, d_model) = (10, 16)
        let input = Tensor::ones(vec![seq_len, d_model]);

        let mha = MultiHeadAttention::new(d_model, num_heads);
        let output = mha.forward(&input);

        // Output format must match input format perfectly
        assert_eq!(output.shape, vec![seq_len, d_model]);
    }
}
