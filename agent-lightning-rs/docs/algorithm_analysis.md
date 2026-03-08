# Từ Điển Bách Khoa Thuật Toán AI: Agent-Roaster

Tài liệu này là cẩm nang chuyên sâu giải phẫu mọi thuật toán Trí Tuệ Nhân Tạo (AI) và Học Máy (Machine Learning) đang được vận hành thực tế bên trong lõi máy `agent-lightning-rs`. Với tiêu chí **Zero-Dependency** (Tự viết tay 100%), bạn có thể trực tiếp theo dõi mã nguồn Rust để đối chiếu với các công thức toán học dưới đây.

---

## Phần 1: Các Thuật Toán Cốt Lõi (Core Engine)

### 1. Multi-Head Self-Attention (MHSA)

- Trọng tâm của kiến trúc **Transformer (LLM)**.
- **Vấn đề giải quyết:** Làm sao để AI hiểu được một "Từ" (hoặc "1 giây rang cà phê") trong ngữ cảnh của toàn bộ chuỗi 100 giây? Làm sao biết khoảng thời gian nào quan trọng nhất?
- **Cách hoạt động:** Cơ chế này tự tạo ra 3 chiếc kính viễn vọng: `Query (Q)`, `Key (K)`, và `Value (V)` thông qua phép nhân ma trận trọng số.
  - `Q` đại diện cho: "Tôi đang tìm kiếm thông tin gì?"
  - `K` đại diện cho: "Tôi chứa thông tin gì?"
  - `V` đại diện cho: "Nội dung thực sự của tôi."
    Điểm chú ý (Attention Score) được tính bằng công thức: $Softmax(\frac{Q \cdot K^T}{\sqrt{d_k}}) \cdot V$
- **File liên quan:** `src/nn/attention.rs`

### 2. GELU (Gaussian Error Linear Unit)

- **Là gì:** Hàm kích hoạt phi tuyến tính (Activation Function). Gần như tất cả LLM hiện đại (GPT-3, Llama, BERT) đều xài cái này thay cho ReLU truyền thống.
- **Ưu điểm:** Khác với ReLU (Cắt phéng mọi số âm thành 0 một cách cộc lốc), GELU giữ lại một chút "linh cảm" số âm bằng xác suất phân phối chuẩn Gauss. Điều này giúp đạo hàm (Gradient) mượt mà hơn, AI học bớt bị "chết" neuron.
- **File liên quan:** `src/core/activation.rs`

### 3. Layer Normalization (LayerNorm)

- **Vấn đề:** Khi dữ liệu đi qua hàng chục tầng Neuron (Deep Learning), các con số có xu hướng phình to quá mức (Exploding) hoặc tiêu biến (Vanishing Gradient).
- **Giải pháp:** LayerNorm sẽ kìm hãm dữ liệu lại sau mỗi tầng mạng. Cụ thể nó sẽ ép điểm `Mean` (Trung bình) của các con số về 0, và `Variance` (Phương sai) về 1. Nhờ thế, mạng Neural hàng trăm lớp vẫn học mượt mà.
- **File liên quan:** `src/nn/normalization.rs`

### 4. Thuật toán Tối ưu Adam (AdamW)

- **Là gì:** Thuật toán dò mìn (Gradient Descent) đỉnh cao nhất thế giới hiện nay.
- **Cách hoạt động:** Thay vì giảm độ dốc mò mẫm vô định (SGD), Adam tính toán dựa trên **Quán tính (Momentum)** của các bước nhảy trước. Nó có 2 bộ nhớ: trí nhớ ngắn (Mean) và trí nhớ dài (Variance). Nhờ độ dốc được "bơm phuộc giảm xóc", mô hình hội tụ cực nhanh (Như ví dụ Loss giảm từ 15000 xuống 5.08 chỉ trong vài Epoch).
- **File liên quan:** `src/core/optimizer.rs`

---

## Phần 2: Thuật Toán Học Tăng Cường (Reinforcement Learning - RL)

### 5. PPO (Proximal Policy Optimization)

- **Là gì:** Thuật toán huấn luyện Agent nổi tiếng do OpenAI sáng lập (dùng để train GPT-1,2,3).
- **Cách hoạt động:** Agent đưa ra phán đoán (Policy Actor), và một vị giám khảo (Critic) sẽ đánh giá hành vi đó tốt hay xấu (Advantage). Đặc biệt, PPO tạo ra cơ chế _Clipping_ (Cắt xén) để ngăn không cho Agent "Học quá nhanh" hoặc "Học lệch" qua những kinh nghiệm nhất thời. Step by step, Agent trưởng thành bền vững.
- **File liên quan:** `src/rl/ppo.rs`

### 6. GRPO (Group Relative Policy Optimization)

- **Là gì:** Thuật toán siêu gọn nhẹ do _DeepSeek_ phát minh. (Mới ra mắt, đang hot toàn cầu hòng thay thế PPO).
- **Cách hoạt động:** Nhược điểm của PPO là tốn RAM (vì cần mạng Critic thứ 2 chấm điểm). GRPO dẹp luôn Critic. Nó bắt Agent tự đưa ra `N` câu trả lời cùng lúc cho một bài toán, sau đó chấm điểm nội bộ giữa các câu trả lời đó (Relative Advantage) dựa trên hàm Reward bằng Toán Học. Tiết kiệm gấp đôi bộ nhớ máy chủ.
- **File liên quan:** `src/rl/grpo.rs`

---

## Hướng dẫn Bắt Đầu Nâng Cao

Để làm chủ toàn bộ hệ thống Agent-Roaster, hãy làm theo lịch trình:

1. Đọc chay cơ chế Forward/Backward của `Tensor` (`src/core/tensor.rs`).
2. Xem cách một Lớp cắt lớp mạng (LinearLayer) và Sequential được móc nối (`src/nn/layer.rs` + `src/nn/network.rs`).
3. Đọc mã nguồn `src/nn/transformer.rs` để thấy Cỗ máy ngôn ngữ được ghép lại từ mảnh MHA, Norm và GELU như thế nào.
4. Chạy `cargo run` với bài thực hành Coffee Dataset (Offline RL / Behavioral Cloning) trong `src/main.rs`.

_Dự án 100% Zero-Dependency - Kiến thức thuộc về bạn._
