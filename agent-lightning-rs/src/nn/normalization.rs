use crate::core::tensor::Tensor;

pub struct LayerNorm {
    pub gamma: Tensor,
    pub beta: Tensor,
    pub eps: f64,
}

impl LayerNorm {
    pub fn new(size: usize) -> Self {
        LayerNorm {
            gamma: Tensor::ones(vec![size]),
            beta: Tensor::zeros(vec![size]),
            eps: 1e-5,
        }
    }

    /// Forward pass cho Layer Normalization
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let n = x.data.len() as f64;
        let mean = x.data.iter().sum::<f64>() / n;

        let var = x.data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let std = (var + self.eps).sqrt();

        let mut out = Tensor::zeros(x.shape.clone());
        for i in 0..x.data.len() {
            let normalized = (x.data[i] - mean) / std;
            out.data[i] = self.gamma.data[i] * normalized + self.beta.data[i];
        }
        out
    }
}
