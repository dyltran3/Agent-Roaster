//! Lồng rang giả lập Vật Lý (Coffee Roaster Environment)
//!
//! Đây là Môi trường học tăng cường (RL Env) để Agent PPO/GRPO tự nướng
//! cà phê ảo hàng triệu lần nhằm tìm ra bộ trọng số tối ưu. Môi trường này
//! tích hợp Physics-Informed Math, EKF, và Reward F-Series.
use crate::envs::state_estimator::ExtendedKalmanFilter;
use crate::rl::credit_assignment::calculate_physics_reward;
use crate::rl::env::{Environment, State, StepResult};

pub struct CoffeeRoasterEnv {
    pub ekf: ExtendedKalmanFilter,
    pub et: f64,           // Environmental Temperature
    pub base_gas: f64,     // Current Gas Setpoint
    pub time_seconds: f64, // Simulation Clock
    pub target_ror: f64,   // Declining ROR profile curve (F-style)
    pub crack_detected: bool,
    pub is_done: bool,
}

impl CoffeeRoasterEnv {
    pub fn new() -> Self {
        CoffeeRoasterEnv {
            ekf: ExtendedKalmanFilter::new(25.0), // Start Room Temp
            et: 25.0,
            base_gas: 10.0,
            time_seconds: 0.0,
            target_ror: 15.0,
            crack_detected: false,
            is_done: false,
        }
    }
}

impl Environment for CoffeeRoasterEnv {
    fn reset(&mut self) -> State {
        self.ekf = ExtendedKalmanFilter::new(25.0);
        self.et = 25.0;
        self.base_gas = 80.0; // Charge gas
        self.time_seconds = 0.0;
        self.target_ror = 15.0; // Maillard phase target
        self.crack_detected = false;
        self.is_done = false;
        self.current_state()
    }

    /// Discrete Action Space Mapping
    /// 0 -> Decrease Residual (-5% Gas)
    /// 1 -> Maintain (0% Gas)
    /// 2 -> Increase Residual (+5% Gas)
    fn step(&mut self, action: usize) -> StepResult {
        if self.is_done {
            return StepResult {
                next_state: self.current_state(),
                reward: 0.0,
                done: true,
                info: "Episode Finished".to_string(),
            };
        }

        let dt = 1.0; // 1 second step
        self.time_seconds += dt;

        let delta_gas = match action {
            0 => -5.0,
            1 => 0.0,
            2 => 5.0,
            _ => 0.0,
        };

        // Hybrid Control limits simulated here
        self.base_gas = (self.base_gas + delta_gas).clamp(0.0, 100.0);

        // Physics 1: Gas heats the drum (ET rises asymptotically to max drum power)
        let drum_max_temp = 300.0 * (self.base_gas / 100.0);
        self.et += 0.02 * (drum_max_temp - self.et) * dt;

        // Physics 2: Generate Noisy BT Sensor Data
        let ideal_bt = self.ekf.x[0] + self.ekf.x[1] * dt;
        let sensor_noise = crate::rand_f64() * 2.0 - 1.0; // +/- 1 degree noise
        let noisy_sensor_bt = ideal_bt + sensor_noise;

        // Physics 3: State Estimator cleans up noise and predicts hidden states
        self.ekf.predict(dt, self.et);
        self.ekf.update(noisy_sensor_bt);

        let t_bean = self.ekf.x[0];
        let ror = self.ekf.x[1];

        // Track target ROR (smooth decay logic for F-13)
        self.target_ror = f64::max(2.0, self.target_ror - 0.0125 * dt); // Slowly lowers over batch

        // Crack Detector (Approx 196°C)
        let was_crack = self.crack_detected;
        if t_bean > 196.0 {
            self.crack_detected = true;
        }
        let just_cracked = !was_crack && self.crack_detected;

        // Physics 4: Calculate PINN-aligned Reward (Reward Shaping)
        let reward = calculate_physics_reward(
            t_bean,
            ror,
            self.target_ror,
            just_cracked,
            self.time_seconds,
            480.0, // Expected crack around 8 minutes (480s)
        );

        // Safety Guard: Stop episode if bounds exceeded
        if t_bean > 230.0 || self.et > 280.0 || self.time_seconds >= 600.0 {
            self.is_done = true;
        }

        StepResult {
            next_state: self.current_state(),
            reward,
            done: self.is_done,
            info: format!("Time: {:.0}s, BT: {:.1}°C", self.time_seconds, t_bean),
        }
    }

    fn action_space(&self) -> usize {
        3 // Actions: 0, 1, 2
    }

    fn state_space(&self) -> usize {
        4 // EKF outputs: [T_bean, ROR, Moisture, CDI]
    }

    fn current_state(&self) -> State {
        vec![self.ekf.x[0], self.ekf.x[1], self.ekf.x[2], self.ekf.x[3]]
    }

    fn name(&self) -> &str {
        "CoffeeRoaster (PINN Edge)"
    }
}
