use crate::core::activation::Activation;
use crate::core::tensor::Tensor;
use crate::nn::attention::MultiHeadAttention;
use crate::nn::network::Sequential;
use crate::nn::normalization::LayerNorm;

/// **Transformer Block**
/// A standard building block for LLMs (like GPT/Llama).
/// Architecture:
///   Input -> LayerNorm -> MHA -> Add (Residual 1)
///   -> LayerNorm -> FeedForward / MLP -> Add (Residual 2)
pub struct TransformerBlock {
    pub attention: MultiHeadAttention,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    pub ffn: Sequential,
}

impl TransformerBlock {
    /// Creates a new Transformer block.
    /// `d_model`: Embedding dimension of the tokens.
    /// `num_heads`: Number of attention heads.
    /// `ffn_hidden`: Hidden dimension of the FeedForward Network (usually 4 * d_model).
    pub fn new(d_model: usize, num_heads: usize, ffn_hidden: usize) -> Self {
        let mut ffn = Sequential::new();
        // Uses GELU activation which is the modern standard for Transformers
        ffn = ffn.dense(d_model, ffn_hidden, Activation::GELU);
        ffn = ffn.dense(ffn_hidden, d_model, Activation::Linear); // Output linear proj

        TransformerBlock {
            attention: MultiHeadAttention::new(d_model, num_heads),
            norm1: LayerNorm::new(d_model),
            norm2: LayerNorm::new(d_model),
            ffn,
        }
    }

    /// Forward pass of the Transformer Block.
    /// Note: Inputs are processed sequentially (Batch=1, Seq_len, d_model)
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // 1. Attention Block with Pre-Normalization and Residual Connection
        let normalized_x = self.norm1.forward(x);
        let attended = self.attention.forward(&normalized_x);
        let residual_1 = x.add(&attended); // x + Attention(Norm(x))

        // 2. FeedForward Block with Pre-Normalization and Residual Connection
        let normalized_res1 = self.norm2.forward(&residual_1);
        let ff_out = self.ffn.forward(&normalized_res1);

        // Final Output = residual_1 + FFN(Norm(residual_1))
        residual_1.add(&ff_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_block_forward_shape() {
        let d_model = 32;
        let num_heads = 4;
        let seq_len = 10;
        let ffn_hidden = d_model * 4;

        // Input sequence: 10 tokens, each embedded in 32 dimensions
        let input = Tensor::ones(vec![seq_len, d_model]);

        let transformer = TransformerBlock::new(d_model, num_heads, ffn_hidden);
        let output = transformer.forward(&input);

        // Output must retain the exact shape [Seq_Len, d_model]
        assert_eq!(output.shape, vec![seq_len, d_model]);
    }
}
