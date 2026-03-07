<div align="center">
  
# ⚡ Agent Lightning RS 🦀
**Khung học tăng cường (RL) phân tán, tốc độ cao, zero-dependency dành cho thiết bị biên (Edge Devices).**

[![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Status: Experimental](https://img.shields.io/badge/Status-Experimental-orange.svg?style=for-the-badge)]()

_Lấy cảm hứng từ cấu trúc Agent Lightning của Microsoft, được xây dựng hoàn toàn từ con số 0 phục vụ cho Edge Computing và Terminal AI._

</div>

---

## 🎯 Giới thiệu

**Agent Lightning RS** là một framework AI/RL mã nguồn mở được viết hoàn toàn bằng **Rust thuần (pure Rust)**, không sử dụng bất kỳ thư viện bên ngoài nào (zero-dependency). Framework được thiết kế chuyên biệt để tối ưu quá trình vận hành mô hình học máy theo hướng **Disaggregated (phân tán)** giữa thiết bị biên (Client/Terminal) và máy chủ huấn luyện (Server).

Với Agent Lightning RS, các thiết bị điện toán yếu vẫn có thể ra quyết định thông minh theo thời gian thực (Real-time AI) nhờ việc tách biệt khâu **Inference (Suy luận)** và **Training (Huấn luyện)** theo kịch bản đồng bộ tốc độ cao.

---

## 🚀 Các tính năng cốt lõi

- 📦 **Zero-dependency (Không phụ thuộc):** Toàn bộ thư viện Tensor, Autograd (đạo hàm tự động), Thuật toán tối ưu (Adam, SGD) và Mạng nơ-ron được tự xây dựng từ đầu 100%. Không cần `tch-rs`, `ndarray` hay `candle`.
- ⚡ **Disaggregated RL Architecture:** Kiến trúc tách biệt Client (thực thi suy luận cực nhẹ) và Server (tập trung sức mạnh tính toán Gradient). Giao tiếp cực nhanh thông qua giao thức mpsc.
- 🧠 **Cơ chế Token-level PPO & GRPO:** Tối ưu hóa điểm thưởng (reward) qua Reward Shaping (AIR) ngay tại runtime. Đặc biệt thiết kế cho các Agent hoạt động theo chuỗi thao tác lý luận (Reasoning Sequence).
- 🧩 **LightningRL (HRL):** Vận hành cấu trúc Học tăng cường Phân cấp (Hierarchical RL) - Agent Quản lý (Manager) tạo mục tiêu trung hạn cho Agent Thực thi (Worker).
- 🎮 **Môi trường Giả lập Tích hợp:** Đi kèm 2 môi trường học tập mẫu: _GridWorld_ (Không gian lưới) và _CartPole_ (Cân bằng gậy xoay).
- 🔌 **Model Context Protocol (MCP):** Sẵn sàng chạy ở chế độ MCP Server, kết nối thẳng vào hệ sinh thái AI Tooling hiện tại.

---

## 📂 Kiến trúc Dự Án

```text
agent-lightning-rs/
├── docs/                      # Tài liệu phân tích Module
├── data/                      # Dataset offline (.csv, JSON)
├── src/
│   ├── core/                  # [TENSORS] Toán học, Autograd, Activations, Losses, Optimizers
│   ├── nn/                    # [NEURAL NET] Layers, Networks, Backprop
│   ├── rl/                    # [RL ALGO] Buffers, PPO, GRPO, Hierarchical, Credit Assignment
│   ├── lightning/             # [DISAGGREGATED] Client, Server, Reward Shaper, Protocol
│   ├── envs/                  # [ENVIRONMENTS] GridWorld, CartPole
│   ├── training/              # [TRAINING] Vòng lặp Loop, Logger, Loader, Checkpointing
│   ├── ui/                    # [ASCII UI] Terminal Dashboard hiển thị Real-time
│   ├── mcp_server.rs          # [MCP] Kết nối giao thức công cụ LLM
│   └── main.rs                # [APP] Entry point để chạy các Demo
├── Cargo.toml                 # [CONFIG] Định nghĩa Project (GNU Toolchain)
└── .gitignore                 # Các tệp bị theo dõi bởi Git
```

---

## 🛠 Hướng dẫn Cài đặt & Chạy lệnh

Do kiến trúc tối ưu hóa cực đoan, code hoàn toàn tự đứng độc lập. Máy tính của bạn chỉ cần **Rust** bản stable mới nhất.

### 1. Chuẩn bị Môi trường (Windowns)

Nếu bạn đang sử dụng Windows và không muốn cài Visual Studio C++ Build Tools nặng nề, hãy sử dụng **GNU Toolchain**:

```bash
rustup toolchain install stable-x86_64-pc-windows-gnu
rustup default stable-x86_64-pc-windows-gnu
```

### 2. Biên dịch & Chạy Huấn Luyện (Demo RL)

Lệnh này sẽ khởi động Dashboard ASCII ngay trong Terminal, tự động chạy qua 3 bài test: **PPO (GridWorld)**, **GRPO (CartPole)**, và **LightningRL (GridWorld)**.

```bash
cargo run --release
```

### 3. Khởi chạy MCP Server (Cho AI Agents)

Nếu bạn muốn dùng LLM tương tác với thư viện RL này qua giao thức kết nối công cụ chuẩn hóa của Anthropic:

```bash
cargo run --bin mcp-server
```

---

## 📊 Terminal Dashboard Training Theo Thời Gian Thực

Khác với các Framework phức tạp yêu cầu thư viện Web/Python (Tensorboard), Agent Lightning RS tích hợp sẵn cơ chế Logger và Chart (Biểu đồ) ngay lập tức trên Terminal bằng ANSI Escape code cực ngầu:

```text
═══════════════════════════════════════════════════════════════
  AGENT LIGHTNING [Training Dashboard]
═══════════════════════════════════════════════════════════════
  Status: Training on GridWorld   | Version: v15
  Episodes: 125                   | Throughput: 153.2 ep/s
  ─────────────────────────────────────────────────────────────
  Current Reward:      50.20 | Mean (last 20):      42.10
  Policy Loss:       0.0416
  ─────────────────────────────────────────────────────────────
  Reward Trend (last 20):
                        ███     ██      ███
                    ████████  ██████  ███████
                  ████████████████████████████████
═══════════════════════════════════════════════════════════════
```

---

## 🤝 Roadmap Phát triển (Sắp ra mắt)

- `[ ]` Khôi phục Checkpointing weights dưới định dạng JSON bên cạnh dạnh nhị phân (Binary) gốc.
- `[ ]` Hoàn thiện **Coffee Dataset Parser** nâng cao để phục vụ dự án Agent-Roaster (Tối ưu điểm rang cà phê bằng AI).
- `[ ]` Thêm môi trường OpenAI Gym thông qua Rust port (nếu cần thiết cho Benchmark).
- `[ ]` Token-level KV Cache cho Autograd hỗ trợ sequence dài hạn cực hạn.

---

## 📜 Giấy Phép

Dự án được phân phối dưới giấy phép **MIT License**. Bạn có thể thoải mái chia sẻ, chỉnh sửa và sử dụng trong nghiên cứu khoa học cũng như ứng dụng thực tế. Xem tệp `LICENSE` để biết thêm chi tiết.
