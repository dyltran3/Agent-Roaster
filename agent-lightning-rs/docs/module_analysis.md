# Module Analysis: Agent Lightning (Rust)

This document provides a technical breakdown of all modules in the Agent Lightning project, their responsibilities, and how they interact to achieve efficient, disaggregated reinforcement learning for terminal devices.

---

## 1. Core Engine (`src/core/`)

The core engine provides the mathematical foundations without external dependencies.

| Module          | Responsibility                                                                    | Real-time Optimization                                                      |
| :-------------- | :-------------------------------------------------------------------------------- | :-------------------------------------------------------------------------- |
| `tensor.rs`     | Primitive multi-dimensional array operations, autograd, and parameter management. | SIMD-ready loops, cache-aligned row-major data, and pre-allocation support. |
| `activation.rs` | Non-linear functions (ReLU, Softmax, Sigmoid).                                    | Vectorized batch processing to reduce function call overhead.               |
| `loss.rs`       | Token-level and sequence-level loss functions (MSE, GRPO, REINFORCE).             | Minimal allocation during backprop.                                         |
| `optimizer.rs`  | Weight update logic (Adam).                                                       | Efficient gradient accumulation.                                            |

## 2. Reinforcement Learning Layer (`src/rl/`)

Implements the Agent Lightning specific algorithms and data structures.

| Module                 | Responsibility                                                                   | Role in Agent Lightning                          |
| :--------------------- | :------------------------------------------------------------------------------- | :----------------------------------------------- |
| `transition.rs`        | Defines the **Unified MDP Transition** (Input, Output, Rewards[], LogProbs[]).   | Foundations for reasoning-sequence optimization. |
| `credit_assignment.rs` | Distributes global rewards into per-token advantages (Uniform, Discounted, GAE). | Step 1 of the LightningRL algorithm.             |
| `lightning_rl.rs`      | Central orchestrator that manages policy and updates.                            | The "Brain" of the training server.              |
| `ppo.rs` / `grpo.rs`   | Algorithm-specific token-level optimization logic.                               | Implement Step 2 (Policy Update).                |
| `buffer.rs`            | Optimized memory for storing recent transitions.                                 | FIFO episode storage.                            |

## 3. Disaggregated Layer (`src/lightning/`)

Facilitates the split between the Agent (Client) and the Trainer (Server).

| Module      | Responsibility                                                   | Key Feature                                     |
| :---------- | :--------------------------------------------------------------- | :---------------------------------------------- |
| `client.rs` | The "Execution" node on the terminal device. Traces transitions. | Lightweight, non-blocking asynchronous sensing. |
| `server.rs` | The "Training" node. Receives traces and emits policy updates.   | Batch processing and versioning.                |
| `reward.rs` | **AIR (Automatic Intermediate Rewarding)** logic.                | Translates tool signals into dense rewards.     |
| `pomdp.rs`  | Shared types and interfaces for the message protocol.            | Zero-copy serialization potential.              |

## 4. Training & Data Layer (`src/training/`)

Handles the lifecycle of a training run.

| Module             | Responsibility                                                        | Configuration                    |
| :----------------- | :-------------------------------------------------------------------- | :------------------------------- |
| `training_loop.rs` | Orchestrates the client-server interaction for local/remote training. | Configurable update intervals.   |
| `dataset.rs`       | Support for offline RL and replay from saved JSON/CSV traces.         | Large-scale experience sampling. |
| `config.rs`        | Hyperparameter management and CLI argument parsing.                   | Centralized control.             |

---

## 5. Architectural Interaction Flow

1. **Client** resets environment and starts an episode.
2. For each step, **Client** performs inference and collects **Unified Transitions**.
3. **RewardShaper (AIR)** adds intermediate signals based on internal state/tool results.
4. **Client** sends `EpisodeDone` with full transition traces to **Server**.
5. **Server** triggers **Credit Assignment** to calculate per-thread advantages.
6. **Server** runs **LightningRL** update (Token-level PPO/GRPO).
7. **Server** increments version and broadcasts updated weights back to **Client**.

## 6. Real-time Design Decisions

- **Zero-Dependency**: Maximizes portability to resource-constrained devices (embedded, edge terminals).
- **Disaggregation**: Offloads heavy tensor computation (training) to a server while keeping the agent footprint minimal.
- **Unified MDP**: Reduces the frequency of server-client communication by grouping reasoning steps.

## 7. Real-time Optimizations Implemented

- **Cache-Optimized MatMul**: RHS matrices are transposed or handled row-wise to ensure linear memory access in inner loops, significantly improving cache hit rates.
- **Allocation-Free Softmax**: Optimized secondary pass across rows to compute exponentials and sums without creating intermediate `Vec<f64>` objects.
- **In-place Linear Forward**: Bypasses general `matmul` matrix multiplication for a direct dot-product implementation that avoids redundant weight transpositions and temporary tensors.
- **Minimal Clone Sequential Loop**: The neural network inference loop was refactored to minimize object ownership transfers and redundant cloning.
- **Zero-Dependency SIMD Potential**: By keeping loops simple and data linear, the Rust compiler (`rustc`) is better able to apply auto-vectorization (SIMD) for modern CPUs on terminal devices.

## 8. Development Environment Troubleshooting

### Windows Linker Issues (`link.exe` not found)

If you encounter `error: linker 'link.exe' not found` on Windows, it means Visual Studio C++ Build Tools are missing. You can either install them or switch to the GNU toolchain:

**Solution (Switch to GNU Toolchain):**

```bash
rustup toolchain install stable-x86_64-pc-windows-gnu
rustup default stable-x86_64-pc-windows-gnu
```
