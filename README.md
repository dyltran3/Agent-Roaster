# ☕ Agent Roaster: Nền Tảng AI Học Tăng Cường & Mô Hình Ngôn Ngữ Xử Lý Hồ Sơ Rang Cà Phê

![License](https://img.shields.io/badge/License-Non--Commercial-red.svg)
![Language](https://img.shields.io/badge/Language-Rust-orange.svg)
![Architecture](https://img.shields.io/badge/Architecture-Transformer%20%7C%20PPO%20%7C%20GRPO-success.svg)
![Dependency](https://img.shields.io/badge/Dependencies-Zero-brightgreen.svg)

**Agent Roaster** là một nền tảng Học Tăng Cường (Reinforcement Learning) và Mô Hình Ngôn Ngữ Lớn (LLM) đột phá được xây dựng hoàn toàn từ con số không (Zero-Dependency) bằng ngôn ngữ **Rust**. Dự án tập trung vào việc mô phỏng, phân tích và tối ưu hóa các hồ sơ rang cà phê chuyên nghiệp thông qua các thuật toán AI tự trị tiên tiến.

---

## 🌟 Chức năng Cốt lõi

Hệ thống kết hợp sức mạnh của lý thuyết Điều khiển tối ưu (MDP) và Trí tuệ ngôn ngữ (Transformer) để tạo ra Agent AI có khả năng tự suy luận và bắt chước chuyên gia.

### 1. Kiến trúc Trí Tuệ (Neural Network & LLM Engine)

Toàn bộ mạng nơ-ron được tự phát triển độc lập không phụ thuộc vào `PyTorch` hay `TensorFlow`.

- **Tensor Core (`src/core/tensor.rs`)**: Lõi tính toán đại số tuyến tính đa chiều (Ma trận, đạo hàm).
- **Transformer Block (`src/nn/transformer.rs`)**: Bộ vi xử lý ngôn ngữ/chuỗi hiện đại ghép nối từ mạng lưới tự chú ý **Multi-Head Self-Attention** (`attention.rs`), **LayerNorm** (`normalization.rs`) và **GELU**.
- **Backpropagation (`src/nn/backprop.rs`)**: Cơ sở tự động tính toán đạo hàm (Auto-Grad) và lan truyền ngược, kết hợp hoàn hảo với **Adam Optimizer** siêu tốc (`optimizer.rs`).

### 2. Hệ thống Học Tăng Cường (Reinforcement Learning)

- **PPO - Proximal Policy Optimization (`src/rl/ppo.rs`)**: Thuật toán OpenAI chuẩn mực có sử dụng _Clipping_ để khống chế tốc độ biến thiên học tập, bảo vệ tác nhân AI khỏi các quyết định "ngớ ngẩn" trong quá trình rang.
- **GRPO - Group Relative Policy Optimization (`src/rl/grpo.rs`)**: Bí quyết thu gọn bộ nhớ mạng Critic mới nhất (DeepSeek type), chấm điểm nội bộ giữa các dự đoán để tìm ra đường cong gia nhiệt (RoR) lý tưởng.

### 3. Hệ thống Giám sát & Bằng chứng Thuật Toán

- Tích hợp ghi nhận trạm dữ liệu theo thời gian thực (Logging Console) màu sắc rõ ràng (GridWorld, Logging).
- Cung cấp tính năng Lưu trữ và Nạp **Magic Header (AGNT)** cho các trọng số weights an toàn (chống Injection Tensor).
- Khả năng **Offline RL / Behavioral Cloning** đỉnh cao. Dự đoán độ lệch Profile chỉ bằng `0.03 °C` so với Master Roaster.

---

## 🚀 Hướng dẫn Cài đặt & Khởi động

Do dự án đạt chuẩn **Zero-Dependency** nên quá trình biên dịch cực kỳ tinh gọn. Chỉ yêu cầu có cài đặt **Rust / Cargo** (>= 1.70).

```bash
# Clone dự án về máy
git clone https://github.com/TuanAnh/Agent-Roaster.git
cd Agent-Roaster/agent-lightning-rs

# Build thư viện tối ưu hóa
cargo build --release

# Chạy hệ thống Đánh giá Toàn cảnh và mô phỏng
cargo run --release
```

---

## 📁 Cấu trúc Dự án

```bash
agent-lightning-rs/
├── docs/                   # Bách Khoa Toàn Thư Algorithm phân tích Toán Học/AI
├── src/
│   ├── core/               # Lõi tính toán Tensor, Loss (MSE/CrossEntropy), Optimizer, Activation (GELU)
│   ├── envs/               # Môi trường mô phỏng (GridWorld, CartPole, Coffee Roasting)
│   ├── lightning/          # Giao thức mạng cho Client/Server, POMDP
│   ├── nn/                 # Kiến trúc Transformer, Attention, Norm, Backprop
│   ├── rl/                 # Thuật toán RL: PPO, GRPO, Multi-Agent (MAPPO)
│   └── training/           # Cơ chế Load/Save Checkpoint và Logging
└── data/                   # Tập dữ liệu mẫu (Standard Roast Profile)
```

---

## 📚 Tài Liệu Bổ Sung & Thuật Toán

Mời bạn tham khảo cẩm nang **Từ Điển AI Đại Cương** được biên soạn bằng Tiếng Việt đính kèm bên trong dự án: `docs/algorithm_analysis.md`. Đây là cánh cửa mở ra toàn bộ thế giới toán học đằng sau Agent-Roaster.

---

## 📜 Giấy Phép (License)

Mã nguồn này được phân phối dưới sự ủy quyền của giấy phép **Non-Commercial License**.
Bạn được phép tự do tùy biến, nghiên cứu và phát triển các hệ thống rang thông minh **cho mục đích học tập cá nhân và phi thương mại**. Mọi hành vi thương mại hóa mã nguồn, đóng gói bán phần mềm hay cung cấp dịch vụ thu tiền dựa trên lõi kiến trúc này đều **bị nghiêm cấm tuyệt đối** nếu không có sự cho phép bằng văn bản từ tác giả. Xem chi tiết tại [LICENSE](LICENSE).
