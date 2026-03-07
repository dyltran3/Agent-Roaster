# Phân tích Module: Agent Lightning (Rust)

Tài liệu này cung cấp bản phân tích kỹ thuật của toàn bộ các module trong dự án Agent Lightning, phân công trách nhiệm của chúng, và cách chúng tương tác để đạt được hiệu năng học tăng cường phân tán (disaggregated reinforcement learning) hiệu quả cho các thiết bị terminal.

---

## 1. Core Engine (Lõi Điện Toán) (`src/core/`)

Core engine cung cấp các nền tảng toán học cơ sở mà không cần bất kỳ dependency (thư viện phụ thuộc) bên ngoài nào.

| Module          | Trách nhiệm (Responsibility)                                                       | Tối ưu hóa Thời gian thực (Real-time Optimization)                                                             |
| :-------------- | :--------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------- |
| `tensor.rs`     | Các phép toán mảng đa chiều cơ sở, autograd (đạo hàm tự động), và quản lý tham số. | Vòng lặp hỗ trợ SIMD, dữ liệu ưu tiên hàng (row-major) chuẩn cache, và hỗ trợ cấp phát trước (pre-allocation). |
| `activation.rs` | Các hàm phi tuyến tính (ReLU, Softmax, Sigmoid).                                   | Xử lý batch dạng vector để giảm cost gọi hàm (function call overhead).                                         |
| `loss.rs`       | Các hàm mất mát cấp độ token và sequence (MSE, GRPO, REINFORCE).                   | Cấp phát tối thiểu (Minimal allocation) trong quá trình backprop.                                              |
| `optimizer.rs`  | Logic cập nhật trọng số (Adam).                                                    | Tích lũy gradient (gradient accumulation) hiệu quả.                                                            |

## 2. Reinforcement Learning Layer (Tầng Học Tăng Cường) (`src/rl/`)

Triển khai các thuật toán và cấu trúc dữ liệu đặc thù của Agent Lightning.

| Module                 | Trách nhiệm (Responsibility)                                                        | Vai trò trong Agent Lightning                                |
| :--------------------- | :---------------------------------------------------------------------------------- | :----------------------------------------------------------- |
| `transition.rs`        | Định nghĩa **Unified MDP Transition** (Input, Output, Rewards[], LogProbs[]).       | Nền tảng cho tối ưu hóa chuỗi suy luận (reasoning-sequence). |
| `credit_assignment.rs` | Phân bổ reward tổng quát thành advantage cho từng token (Uniform, Discounted, GAE). | Bước 1 của thuật toán LightningRL.                           |
| `lightning_rl.rs`      | Bộ điều phối trung tâm (Central orchestrator) quản lý policy và vòng cập nhật.      | Là bộ não ("Brain") của training server.                     |
| `ppo.rs` / `grpo.rs`   | Logic tối ưu cấp độ token cho thuật toán cụ thể.                                    | Triển khai Bước 2 (Cập nhật Policy).                         |
| `buffer.rs`            | Bộ nhớ tối ưu để lưu trữ các transition gần nhất.                                   | Lưu trữ episode theo chuẩn FIFO.                             |

## 3. Disaggregated Layer (Tầng Phân Tán) (`src/lightning/`)

Hỗ trợ việc phân tách giữa Agent (Client) và Trainer (Server).

| Module      | Trách nhiệm (Responsibility)                                                     | Tính năng chính (Key Feature)                            |
| :---------- | :------------------------------------------------------------------------------- | :------------------------------------------------------- |
| `client.rs` | Node "Execution" (thực thi) trên thiết bị terminal. Lấy dấu (trace) transitions. | Bắt tín hiệu bất đồng bộ, siêu nhẹ, non-blocking.        |
| `server.rs` | Node "Training" (huấn luyện). Nhận các trace và phát đi bản cập nhật policy.     | Xử lý theo batch và quản lý version.                     |
| `reward.rs` | Logic **AIR (Automatic Intermediate Rewarding)**.                                | Chuyển đổi tín hiệu hệ thống (tool) thành dense rewards. |
| `pomdp.rs`  | Các type dùng chung và interface cho giao thức tin nhắn (message protocol).      | Tiềm năng serialization theo chuẩn zero-copy.            |

## 4. Training & Data Layer (Tầng Dữ Liệu & Huấn Luyện) (`src/training/`)

Quản lý chu kỳ phân tách (lifecycle) của một phiên huấn luyện (training run).

| Module             | Trách nhiệm (Responsibility)                                   | Cấu hình (Configuration)          |
| :----------------- | :------------------------------------------------------------- | :-------------------------------- |
| `training_loop.rs` | Điều phối tương tác client-server cho local/remote training.   | Thông số khoảng thời gian update. |
| `dataset.rs`       | Hỗ trợ offline RL và replay/lặp lại từ các tệp trace JSON/CSV. | Sampling kinh nghiệm quy mô lớn.  |
| `config.rs`        | Quản lý siêu tham số (Hyperparameter) và phân tích đối số CLI. | Quản lý tập trung.                |

---

## 5. Architectural Interaction Flow (Luồng Tương Tác Kiến Trúc)

1. **Client** gọi reset môi trường và bắt đầu một episode.
2. Với mỗi bước tính (step), **Client** làm inference và thu thập các **Unified Transitions**.
3. **RewardShaper (AIR)** bổ sung các tín hiệu trung gian dựa trên trạng thái nội tại / kết quả của tool.
4. **Client** gửi tín hiệu `EpisodeDone` kèm toàn bộ log trace của transition về cho **Server**.
5. **Server** kích hoạt **Credit Assignment** để tính toán advantage cho từng luồng (thread).
6. **Server** chạy bộ cập nhật **LightningRL** (Token-level PPO/GRPO).
7. **Server** tăng version lên 1 bậc và phát sóng (broadcast) các trọng số (weights) đã cập nhật trả về cho **Client**.

## 6. Real-time Design Decisions (Quyết Định Thiết Kế Dành Cho Thời Gian Thực)

- **Zero-Dependency**: Tối đa hóa khả năng di động (portability) cho phần mềm để chạy trên các thiết bị hạn chế tài nguyên (nhúng, edge terminals).
- **Disaggregation (Phân tán)**: Offload/Chuyển giao việc tính toán tensor nặng nhọc (training) lên một máy chủ trong khi vẫn giữ lại cực kỳ ít footprints cho agent.
- **Unified MDP**: Giảm tần suất giao tiếp server-client bằng cách gộp nhóm (grouping) các chuỗi suy luận (reasoning steps).

## 7. Real-time Optimizations Implemented (Các Tối Ưu Hóa Tốc Độ Đã Đưa Vào)

- **Cache-Optimized MatMul**: Các ma trận phía bên phải (RHS) được transpose/chuyển vị hoặc xử lý theo luồng hàng (row-wise) để đảm bảo bộ nhớ tuyến tính trong inner loop, góp phần cải thiện vượt trội tỷ lệ cache hit.
- **Allocation-Free Softmax**: Lượt xử lý thứ hai (secondary pass) tối ưu qua các hàng tính toán hàm số mũ (exponentials) và tổng mà không cần khởi tạo thêm các đối tượng `Vec<f64>` trung gian.
- **In-place Linear Forward**: Vượt qua chu trình nhân ma trận chung (general matmul) bằng một thuật toán tính tích vô hướng trực tiếp (dot-product) tránh mọi vòng chuyển vị tham số thừa hay sử dụng tensor tạm.
- **Minimal Clone Sequential Loop**: Neural network inference loop được cấu trúc lại nhằm giảm thiểu tối đa các yêu cầu chuyển nhượng/clone object trong runtime.
- **Zero-Dependency SIMD Potential**: Thông qua tư duy giữ các vòng lặp xử lý dữ liệu đơn giản và tuyến tính (linear), trình compiler (`rustc`) có năng lực mạnh mẽ hơn để kích hoạt auto-vectorization (SIMD) trên cả những dòng CPU hiện đại của terminal devices.

## 8. Development Environment Troubleshooting (Hướng Dẫn Sửa Lỗi Môi Trường Compiler)

### Windows Linker Issues (Không tìm thấy `link.exe`)

Nếu bạn bắt gặp báo lỗi `error: linker 'link.exe' not found` trên hệ điều hành Windows, có nghĩa là bộ Visual Studio C++ Build Tools đang bị mất. Bạn có thể hoặc chọn cách tải chúng về, hoặc chuyển đổi Rust sang toolchain cấu hình của GNU:

**Giải pháp (Chuyển đổi sang GNU Toolchain):**

```bash
rustup toolchain install stable-x86_64-pc-windows-gnu
rustup default stable-x86_64-pc-windows-gnu
```
