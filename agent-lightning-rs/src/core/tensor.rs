/// Tensor Engine — viết thuần Rust, không dùng bất kỳ ML library nào.
///
/// Hỗ trợ:
///   - Tensor 1D / 2D với shape bất kỳ
///   - Các phép: add, sub, mul, div (element-wise), matmul, transpose
///   - Gradient tracking (requires_grad, grad field)
///   - Broadcast cơ bản (scalar broadcasting)
use std::fmt;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
    pub requires_grad: bool,
    pub grad: Option<Vec<f64>>,
}

impl Tensor {
    // --- Constructors ---

    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected,
            "Data length {} doesn't match shape {:?} (expected {})",
            data.len(),
            shape,
            expected
        );
        Tensor {
            data,
            shape,
            requires_grad: false,
            grad: None,
        }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Tensor::new(vec![0.0; size], shape)
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Tensor::new(vec![1.0; size], shape)
    }

    pub fn from_scalar(val: f64) -> Self {
        Tensor::new(vec![val], vec![1])
    }

    /// Random normal initialization (Box-Muller transform)
    pub fn randn(shape: Vec<usize>, mean: f64, std: f64) -> Self {
        use std::f64::consts::PI;
        let size: usize = shape.iter().product();
        let mut data = Vec::with_capacity(size);
        let mut i = 0;
        while data.len() < size {
            // Use time-seeded LCG for portability (no rand dependency in core tensor)
            let u1 = lcg_rand(i as u64) as f64 / u64::MAX as f64;
            let u2 = lcg_rand(i as u64 + 1) as f64 / u64::MAX as f64;
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).sin();
            data.push(mean + std * z0);
            if data.len() < size {
                data.push(mean + std * z1);
            }
            i += 2;
        }
        Tensor::new(data, shape)
    }

    /// Xavier/Glorot initialization — recommended for deep networks
    pub fn xavier(rows: usize, cols: usize) -> Self {
        let scale = (6.0_f64 / (rows + cols) as f64).sqrt();
        Self::randn(vec![rows, cols], 0.0, scale)
    }

    // --- Properties ---

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn is_scalar(&self) -> bool {
        self.data.len() == 1
    }

    /// Get shape as (rows, cols) for 2D tensors
    pub fn shape2d(&self) -> (usize, usize) {
        assert_eq!(
            self.shape.len(),
            2,
            "Expected 2D tensor, got shape {:?}",
            self.shape
        );
        (self.shape[0], self.shape[1])
    }

    // --- Indexing ---

    pub fn get(&self, indices: &[usize]) -> f64 {
        let idx = self.flat_index(indices);
        self.data[idx]
    }

    pub fn set(&mut self, indices: &[usize], val: f64) {
        let idx = self.flat_index(indices);
        self.data[idx] = val;
    }

    fn flat_index(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.shape.len());
        let mut idx = 0;
        let mut stride = 1;
        for (i, &dim) in self.shape.iter().enumerate().rev() {
            idx += indices[i] * stride;
            stride *= dim;
        }
        idx
    }

    // --- Element-wise operations ---

    pub fn add(&self, other: &Tensor) -> Tensor {
        if other.is_scalar() {
            return self.add_scalar(other.data[0]);
        }
        assert_eq!(
            self.shape, other.shape,
            "Shape mismatch in add: {:?} vs {:?}",
            self.shape, other.shape
        );
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a + b)
            .collect();
        Tensor::new(data, self.shape.clone())
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        if other.is_scalar() {
            return self.add_scalar(-other.data[0]);
        }
        assert_eq!(self.shape, other.shape, "Shape mismatch in sub");
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a - b)
            .collect();
        Tensor::new(data, self.shape.clone())
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        if other.is_scalar() {
            return self.mul_scalar(other.data[0]);
        }
        assert_eq!(self.shape, other.shape, "Shape mismatch in mul");
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a * b)
            .collect();
        Tensor::new(data, self.shape.clone())
    }

    pub fn div(&self, other: &Tensor) -> Tensor {
        if other.is_scalar() {
            return self.mul_scalar(1.0 / other.data[0]);
        }
        assert_eq!(self.shape, other.shape, "Shape mismatch in div");
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a / b)
            .collect();
        Tensor::new(data, self.shape.clone())
    }

    pub fn add_scalar(&self, s: f64) -> Tensor {
        Tensor::new(
            self.data.iter().map(|x| x + s).collect(),
            self.shape.clone(),
        )
    }

    pub fn mul_scalar(&self, s: f64) -> Tensor {
        Tensor::new(
            self.data.iter().map(|x| x * s).collect(),
            self.shape.clone(),
        )
    }

    pub fn pow(&self, exp: f64) -> Tensor {
        Tensor::new(
            self.data.iter().map(|x| x.powf(exp)).collect(),
            self.shape.clone(),
        )
    }

    pub fn sqrt(&self) -> Tensor {
        Tensor::new(
            self.data.iter().map(|x| x.sqrt()).collect(),
            self.shape.clone(),
        )
    }

    pub fn abs(&self) -> Tensor {
        Tensor::new(
            self.data.iter().map(|x| x.abs()).collect(),
            self.shape.clone(),
        )
    }

    pub fn neg(&self) -> Tensor {
        self.mul_scalar(-1.0)
    }

    pub fn exp(&self) -> Tensor {
        Tensor::new(
            self.data.iter().map(|x| x.exp()).collect(),
            self.shape.clone(),
        )
    }

    pub fn ln(&self) -> Tensor {
        Tensor::new(
            self.data.iter().map(|x| x.ln()).collect(),
            self.shape.clone(),
        )
    }

    pub fn clamp(&self, min: f64, max: f64) -> Tensor {
        Tensor::new(
            self.data.iter().map(|x| x.clamp(min, max)).collect(),
            self.shape.clone(),
        )
    }

    // --- Reduction operations ---

    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    pub fn mean(&self) -> f64 {
        self.sum() / self.data.len() as f64
    }

    pub fn max(&self) -> f64 {
        self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    pub fn min(&self) -> f64 {
        self.data.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    pub fn argmax(&self) -> usize {
        self.data
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }

    /// Sum along axis 0 (row-wise collapse for 2D)
    pub fn sum_axis0(&self) -> Tensor {
        let (rows, cols) = self.shape2d();
        let mut result = vec![0.0f64; cols];
        for r in 0..rows {
            for (c, item) in result.iter_mut().enumerate().take(cols) {
                *item += self.data[r * cols + c];
            }
        }
        Tensor::new(result, vec![1, cols])
    }

    // --- Matrix operations ---

    /// Matrix multiply: (m×k) @ (k×n) → (m×n)
    /// Optimized for cache locality by transposing the RHS matrix first.
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let (m, k) = self.shape2d();
        let (k2, n) = other.shape2d();
        assert_eq!(
            k, k2,
            "matmul shape mismatch: [{},{}] @ [{},{}]",
            m, k, k2, n
        );

        // Optimization: Transpose `other` to (n, k) so we can do linear dot products
        // This significantly improves cache performance for the inner loop.
        let other_t = other.transpose();
        let mut result = vec![0.0f64; m * n];

        for i in 0..m {
            let row_offset = i * k;
            let self_row = &self.data[row_offset..row_offset + k];

            for j in 0..n {
                let other_row_offset = j * k;
                let other_col = &other_t.data[other_row_offset..other_row_offset + k];

                // Auto-vectorization / SIMD iterators
                let sum: f64 = self_row
                    .iter()
                    .zip(other_col.iter())
                    .map(|(a, b)| a * b)
                    .sum();

                result[i * n + j] = sum;
            }
        }
        Tensor::new(result, vec![m, n])
    }

    /// Transpose 2D tensor: (m×n) → (n×m)
    pub fn transpose(&self) -> Tensor {
        let (rows, cols) = self.shape2d();
        let mut result = vec![0.0f64; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                result[c * rows + r] = self.data[r * cols + c];
            }
        }
        Tensor::new(result, vec![cols, rows])
    }

    /// Concatenate two 2D tensors along axis 0
    pub fn concat_axis0(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 2);
        assert_eq!(other.shape.len(), 2);
        assert_eq!(self.shape[1], other.shape[1], "Dim 1 must match for concat");

        let mut new_data = self.data.clone();
        new_data.extend_from_slice(&other.data);

        Tensor::new(
            new_data,
            vec![self.shape[0] + other.shape[0], self.shape[1]],
        )
    }

    /// Reshape tensor (total elements must match)
    pub fn reshape(&self, new_shape: Vec<usize>) -> Tensor {
        let new_size: usize = new_shape.iter().product();
        assert_eq!(
            self.data.len(),
            new_size,
            "Cannot reshape {:?} to {:?}",
            self.shape,
            new_shape
        );
        Tensor::new(self.data.clone(), new_shape)
    }

    /// Flatten to 1D
    pub fn flatten(&self) -> Tensor {
        let len = self.data.len();
        Tensor::new(self.data.clone(), vec![len])
    }

    /// Stack vector of 1D tensors into a 2D matrix (batch × features)
    pub fn stack_rows(rows: &[Tensor]) -> Tensor {
        assert!(!rows.is_empty(), "Cannot stack empty slice");
        let cols = rows[0].data.len();
        let mut data = Vec::with_capacity(rows.len() * cols);
        for row in rows {
            assert_eq!(row.data.len(), cols, "All rows must have same length");
            data.extend_from_slice(&row.data);
        }
        Tensor::new(data, vec![rows.len(), cols])
    }

    /// Normalize: (x - mean) / (std + eps)
    pub fn normalize(&self, eps: f64) -> Tensor {
        let mean = self.mean();
        let variance =
            self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / self.data.len() as f64;
        let std = (variance + eps).sqrt();
        self.add_scalar(-mean).mul_scalar(1.0 / std)
    }

    // --- Gradient support ---

    pub fn with_grad(mut self) -> Self {
        self.requires_grad = true;
        self.grad = Some(vec![0.0; self.data.len()]);
        self
    }

    pub fn zero_grad(&mut self) {
        if let Some(ref mut g) = self.grad {
            for x in g.iter_mut() {
                *x = 0.0;
            }
        }
    }

    pub fn accumulate_grad(&mut self, grad: &[f64]) {
        assert_eq!(self.data.len(), grad.len());
        if self.grad.is_none() {
            self.grad = Some(vec![0.0; self.data.len()]);
        }
        if let Some(ref mut g) = self.grad {
            for (g_i, dg_i) in g.iter_mut().zip(grad.iter()) {
                *g_i += dg_i;
            }
        }
    }

    pub fn apply_gradient(&mut self, lr: f64) {
        if let Some(ref g) = self.grad.clone() {
            for (w, dw) in self.data.iter_mut().zip(g.iter()) {
                *w -= lr * dw;
            }
        }
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape={:?}, data=[", self.shape)?;
        let preview: Vec<String> = self
            .data
            .iter()
            .take(8)
            .map(|x| format!("{:.4}", x))
            .collect();
        write!(f, "{}", preview.join(", "))?;
        if self.data.len() > 8 {
            write!(f, ", ...")?;
        }
        write!(f, "])")
    }
}

/// Simple LCG random number generator (no external dependency)
fn lcg_rand(seed: u64) -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as u64;
    let s = seed.wrapping_add(t).wrapping_add(0x9e3779b97f4a7c15);
    let mut z = s;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

// --- Edge AI Optimizations (Memory & Physics) ---

/// Arena Memory Pool to avoid Heap fragmentation on Edge devices
pub struct TensorArena {
    pub buffer: Vec<f64>,
    pub offset: usize,
}

impl TensorArena {
    pub fn new(capacity: usize) -> Self {
        TensorArena {
            buffer: vec![0.0; capacity],
            offset: 0,
        }
    }

    pub fn allocate(&mut self, size: usize) -> &[f64] {
        if self.offset + size > self.buffer.len() {
            panic!("Arena Out of Memory");
        }
        let start = self.offset;
        self.offset += size;
        &self.buffer[start..self.offset]
    }

    pub fn reset(&mut self) {
        self.offset = 0;
    }
}

/// KV Cache for Transformer models (Edge AI Memory Optimization)
#[derive(Clone, Debug)]
pub struct KVCache {
    pub k_cache: Option<Tensor>,
    pub v_cache: Option<Tensor>,
}

impl KVCache {
    pub fn new() -> Self {
        KVCache {
            k_cache: None,
            v_cache: None,
        }
    }

    pub fn update(&mut self, new_k: &Tensor, new_v: &Tensor) -> (Tensor, Tensor) {
        let k = match &self.k_cache {
            Some(curr_k) => curr_k.concat_axis0(new_k),
            None => new_k.clone(),
        };
        let v = match &self.v_cache {
            Some(curr_v) => curr_v.concat_axis0(new_v),
            None => new_v.clone(),
        };
        self.k_cache = Some(k.clone());
        self.v_cache = Some(v.clone());
        (k, v)
    }
}

/// Simple int8 Quantization for storage
pub fn quantize_f64_to_i8(data: &[f64]) -> (Vec<i8>, f64) {
    let max_val = data.iter().fold(0.0f64, |acc, &x| acc.max(x.abs()));
    let scale = if max_val > 0.0 { 127.0 / max_val } else { 1.0 };
    let quantized = data.iter().map(|&x| (x * scale).round() as i8).collect();
    (quantized, scale)
}

pub fn dequantize_i8_to_f64(data: &[i8], scale: f64) -> Vec<f64> {
    data.iter().map(|&x| (x as f64) / scale).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let t = Tensor::zeros(vec![3, 4]);
        assert_eq!(t.numel(), 12);
        assert_eq!(t.shape, vec![3, 4]);
    }

    #[test]
    fn test_add() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
        let c = a.add(&b);
        assert_eq!(c.data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_matmul() {
        // [2×3] @ [3×2] = [2×2]
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
        let c = a.matmul(&b);
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data[0], 58.0); // 1*7 + 2*9 + 3*11
        assert_eq!(c.data[3], 136.0); // 4*8 + 5*10 + 6*12
    }

    #[test]
    fn test_transpose() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = a.transpose();
        assert_eq!(b.shape, vec![3, 2]);
        assert_eq!(b.data[0], 1.0);
        assert_eq!(b.data[1], 4.0);
    }

    #[test]
    fn test_normalize() {
        let t = Tensor::new(vec![2.0, 4.0, 6.0, 8.0], vec![4]);
        let n = t.normalize(1e-8);
        assert!((n.mean()).abs() < 1e-6);
    }
}
