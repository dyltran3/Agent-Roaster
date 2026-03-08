# Phân tích Module: Agent Lightning (Rust)

Tài liệu này cung cấp bản phân tích kỹ thuật của toàn bộ các module trong dự án Agent Lightning, phân công trách nhiệm của chúng, và cách chúng tương tác để đạt được hiệu quả học tăng cường phân tán (disaggregated reinforcement learning) cho các thiết bị terminal.

---

## 1. Core Engine (Lõi Điện Toán) - `src/core/`

Core engine cung cấp các nền tảng toán học cơ sở hoàn toàn độc lập mà không cần bất kỳ dependency (thư viện phụ thuộc) bên ngoài nào.

- 📐 **`tensor.rs`**
  - **Trách nhiệm:** Triển khai các phép toán mảng đa chiều cơ sở, autograd (đạo hàm tự động), và quản lý rẽ nhánh tham số.
  - **Tối ưu hóa (Real-time):** Vòng lặp hỗ trợ SIMD, bố trí dữ liệu ưu tiên hàng (row-major) để tăng độ trúng đích của cache, và hỗ trợ cấp phát bộ nhớ trước (pre-allocation).

- ⚡ **`activation.rs`**
  - **Trách nhiệm:** Các hàm kích hoạt phi tuyến tính (ReLU, Softmax, Sigmoid,...).
  - **Tối ưu hóa (Real-time):** Xử lý batch dạng vector song song để giảm chi phí gọi hàm (function call overhead).

- 📉 **`loss.rs`**
  - **Trách nhiệm:** Các hàm tính toán sự mất mát cấp độ token và sequence (như MSE, độ lệch GRPO, REINFORCE).
  - **Tối ưu hóa (Real-time):** Giam giữ việc cấp phát bộ nhớ ở mức độ tối thiểu (Minimal allocation) trong suốt quá trình backprop.

- ⚙️ **`optimizer.rs`**
  - **Trách nhiệm:** Tính toán logic cập nhật trọng số (hiện tại hỗ trợ Adam, SGD).
  - **Tối ưu hóa (Real-time):** Tích lũy gradient (gradient accumulation) cực kỳ hiệu quả.

---

## 2. Reinforcement Learning Layer (Tầng Học Tăng Cường) - `src/rl/`

Nơi triển khai các thuật toán và cấu trúc dữ liệu đặc thù mang bản sắc của Agent Lightning.

- 🔄 **`transition.rs`**
  - **Trách nhiệm:** Định nghĩa cấu trúc **Unified MDP Transition** (bao gồm Input, Output, Rewards, LogProbs).
  - **Vai trò hệ thống:** Đóng vai trò là nền tảng nguyên tử cho việc tối ưu hóa theo chuỗi suy luận (reasoning-sequence).

- 🏆 **`credit_assignment.rs`**
  - **Trách nhiệm:** Phân bổ phần thưởng tổng quát thành các advantage (lợi thế) cho từng token sinh ra (hỗ trợ Uniform, Discounted, GAE).
  - **Vai trò hệ thống:** Chịu trách nhiệm hoàn thành Bước 1 của thuật toán cốt lõi LightningRL.

- 🧠 **`lightning_rl.rs`**
  - **Trách nhiệm:** Sự thay thế cho Bộ điều phối trung tâm (Central orchestrator) - làm nhiệm vụ quản lý policy và phân tách vòng cập nhật.
  - **Vai trò hệ thống:** Được ví như là bộ não ("Brain") của hệ thống training server.

- 🎯 **`ppo.rs` / `grpo.rs`**
  - **Trách nhiệm:** Logic tối ưu thuật toán cấp độ point-token.
  - **Vai trò hệ thống:** Bản mô tả thi công thực tế của Bước 2 (Cập nhật Policy dựa trên Advantage).

- 💾 **`buffer.rs`**
  - **Trách nhiệm:** Cung cấp bộ nhớ cấu trúc tối ưu để lưu trữ các transition gần nhất tạm thời.
  - **Vai trò hệ thống:** Nơi lưu trữ episode theo chuẩn vào-trước-ra-trước (FIFO) cho việc bốc mẫu mini-batch.

---

## 3. Disaggregated Layer (Tầng Phân Tán) - `src/lightning/`

Công cụ hỗ trợ việc phân tách ranh giới và tạo giao thức nối tiếp giữa Agent (ở vai trò Client) và Trainer (ở vai trò Server).

- 🎒 **`client.rs`**
  - **Trách nhiệm:** Hoạt động như Node "Execution" (thực thi) trên thiết bị terminal. Nó chỉ có việc suy luận và lấy dấu (trace) transitions.
  - **Tính năng chính:** Vận hành bắt tín hiệu bất đồng bộ, siêu nhẹ, không làm nghẽn luồng (non-blocking).

- 🖥️ **`server.rs`**
  - **Trách nhiệm:** Hoạt động như Node "Training" (huấn luyện). Chờ đợi nhận các trace, tính toán ngược, và định kì phát đi bản cập nhật policy.
  - **Tính năng chính:** Xử lý dữ liệu gộp theo batch lớn và quản lý vòng đời version của model.

- 🎁 **`reward.rs`**
  - **Trách nhiệm:** Chứa cấu trúc logic **AIR (Automatic Intermediate Rewarding)** do Agent Lightning đề xuất.
  - **Tính năng chính:** Chuyển đổi các tín hiệu rời rạc của hệ thống (như Tool Calls) thành các dense rewards có độ trù phú cao.

- 📡 **`pomdp.rs`**
  - **Trách nhiệm:** Định nghĩa các type dùng chung và abstraction interface cho giao thức tin nhắn (message protocol) giữa Client-Server.
  - **Tính năng chính:** Chuẩn bị tiềm năng cho việc serialization theo chuỗi zero-copy hiệu năng cao.

---

## 4. Training & Data Layer (Tầng Dữ Liệu & Huấn Luyện) - `src/training/`

Quản lý tất cả chu kỳ phân tách vòng đời (lifecycle) của một lệnh vận hành AI huấn luyện.

- ♾️ **`training_loop.rs`**
  - **Trách nhiệm:** Điều phối trực tiếp tương tác client-server tại local terminal theo mô hình đúc kết kết quả liên tục.
  - **Cấu hình:** Điều chỉnh các thông số khoảng thời gian update thông minh.

- 📁 **`dataset.rs`**
  - **Trách nhiệm:** Abstract hỗ trợ offline RL và replay/lặp lại từ các tệp trace JSON/CSV thô.
  - **Cấu hình:** Sampling lại kinh nghiệm quy mô lớn giúp mạng học nhanh hơn.

- 🎛️ **`config.rs`**
  - **Trách nhiệm:** Quản lý toàn bộ thông số đầu vào siêu tham số (Hyperparameter) và phân tích các đối số gọi qua CLI.
  - **Cấu hình:** Làm "Control Panel" quản lý thông tin tập trung.

---

## 5. Architectural Interaction Flow (Luồng Tương Tác Kiến Trúc)

1. **Client** gọi tác vụ làm sạch môi trường (reset) và bắt đầu một episode mới.
2. Với mỗi bước tính diễn ra (step), **Client** sẽ làm inference sinh text và thu thập các **Unified Transitions**.
3. **RewardShaper (AIR)** lặng lẽ quan sát và bổ sung các tín hiệu trung gian dựa trên trạng thái nội tại / kết quả trả về của các tool.
4. Cuối episode, **Client** gửi nhãn `EpisodeDone` kèm toàn bộ log trace của transition về cho quá trình **Server**.
5. **Server** nhận được, kích hoạt **Credit Assignment** để tính toán điểm lợi thế advantage tách biệt cho từng luồng (thread).
6. **Server** tiếp tục chạy bộ cập nhật **LightningRL** (Token-level PPO/GRPO) để đẩy model tiến thêm một bước tối ưu hiệu điện thế loss.
7. **Server** tăng hệ version lên 1 bậc và phát sóng (broadcast) các trọng số (weights) đã cập nhật thẳng về bộ nhớ của **Client**. Quá trình lặp lại.

## 6. Real-time Design Decisions (Quyết Định Thiết Kế Dành Cho Thời Gian Thực)

- **Zero-Dependency**: Tối đa hóa khả năng vận hành trơn tru di động (portability) cho phần mềm để sẵn sàng chạy trên các thiết bị giới hạn lượng RAM, CPU (Edge terminals).
- **Disaggregation (Phân tán)**: Offload phần tính toán lượng giác tensor nặng nhọc (training) lên một máy chủ mạng nội bộ / cloud mạnh mẽ, trong khi vẫn giữ nguyên trạng vô cùng bé (footprints) cho agent cục bộ thao tác.
- **Unified MDP**: Nhóm (group) các chuỗi suy luận (reasoning steps) lại để đóng gói vào các payload chung rồi gửi đi, cắt giảm triệt để số lượng kết nối mạng thừa giữa Server/Client.

## 7. Real-time Optimizations Implemented (Các Tối Ưu Hóa Tối Đa Tốc Độ Đã Được Đưa Vào Code)

- **Cache-Optimized MatMul**: Khi tiến hành phép nhân ma trận, các ma trận phía bên phải (RHS) đều được ép buộc transpose (chuyển vị) hoặc xử lý theo luồng hàng (row-wise) nhằm đảm bảo layout bộ nhớ tuyến tính trong inner loop. Thao tác này kéo tốc độ tính toán lên gấp bội do tránh được hiện tượng Cache Miss của CPU.
- **Allocation-Free Softmax**: Lượt xử lý thứ hai (secondary pass) duyệt qua các hàng tính toán hàm số mũ (exponentials) và lấy hàm chia tổng đã được thực hiện bằng Iterator mà không cần khởi tạo lấy bộ nhớ heap sinh các đối tượng dạng `Vec<f64>` trung gian.
- **In-place Linear Forward**: Với tác vụ feed-forward cơ bản, framework sẽ bỏ qua chu trình nhân ma trận chung (general matmul) để chạy qua một đoạn code tích vô hướng trực tiếp (dot-product) tránh mọi vòng lặp chuyển vị tham số bộ nhớ thừa.
- **Minimal Clone Sequential Loop**: Neural network inference lúc chạy Sequential layers được cấu trúc lại sử dụng Lifetimes `&` và reference thay vì yêu cầu chuyển nhượng / Clone object vô cớ.
- **Zero-Dependency SIMD Potential**: Thông qua triết lý giữ các vòng lặp xử lý dữ liệu đơn điệu, cơ bản và phẳng (linear), trình compiler (`rustc`) mặc nhiên cung cấp năng lực tự động áp dụng vectorization (SIMD) trên các bộ não CPU hiện đại và vi kiến trúc ARM/x86.

## 8. Development Environment Troubleshooting (Hướng Dẫn Sửa Lỗi Tương Thích Complier)

### Windows Linker Issues (Không tìm thấy `link.exe`)

Nếu bạn gọi lệnh build và gặp báo lỗi `error: linker 'link.exe' not found` trên hệ điều hành Windows, có nghĩa là bộ Visual Studio C++ Build Tools đang bị trống khỏi hệ thống. Bạn có thể cài đặt chúng, hoặc nhanh gọn hơn là chuyển đổi Rust sang tích hợp toolchain GNU thuần C:

**Giải pháp (Chuyển đổi sang GNU Toolchain):**

```bash
rustup toolchain install stable-x86_64-pc-windows-gnu
rustup default stable-x86_64-pc-windows-gnu
```
