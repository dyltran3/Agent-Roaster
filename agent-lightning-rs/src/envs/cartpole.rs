/// CartPole — classic control problem for testing RL algorithms.
///
/// A pole is attached to a cart moving on a frictionless track.
/// The agent must keep the pole balanced by pushing the cart left or right.
///
/// State:  (position, velocity, pole_angle, pole_angular_velocity) — 4 features
/// Action: 0=Push Left, 1=Push Right
/// Reward: +1 for every step the pole stays upright
use crate::rl::env::{Action, Environment, State, StepResult};
use std::f64::consts::PI;

const GRAVITY: f64 = 9.8;
const CART_MASS: f64 = 1.0;
const POLE_MASS: f64 = 0.1;
const POLE_HALF_LEN: f64 = 0.5;
const FORCE_MAG: f64 = 10.0;
const DT: f64 = 0.02; // timestep (seconds)
const MAX_ANGLE: f64 = 12.0 * PI / 180.0; // 12 degrees
const MAX_POS: f64 = 2.4;
const MAX_STEPS: usize = 500;

pub struct CartPole {
    pub position: f64,
    pub velocity: f64,
    pub angle: f64,
    pub ang_velocity: f64,
    pub steps: usize,
}

impl CartPole {
    pub fn new() -> Self {
        CartPole {
            position: 0.0,
            velocity: 0.0,
            angle: 0.05,
            ang_velocity: 0.0,
            steps: 0,
        }
    }

    fn make_state(&self) -> State {
        // Normalize to reasonable range for NN input
        vec![
            self.position / MAX_POS,
            self.velocity.clamp(-5.0, 5.0) / 5.0,
            self.angle / MAX_ANGLE,
            self.ang_velocity.clamp(-5.0, 5.0) / 5.0,
        ]
    }

    fn physics_step(&mut self, force: f64) {
        let total_mass = CART_MASS + POLE_MASS;
        let cos_a = self.angle.cos();
        let sin_a = self.angle.sin();
        let pole_mass_len = POLE_MASS * POLE_HALF_LEN;

        let temp = (force + pole_mass_len * self.ang_velocity.powi(2) * sin_a) / total_mass;
        let ang_acc = (GRAVITY * sin_a - cos_a * temp)
            / (POLE_HALF_LEN * (4.0 / 3.0 - POLE_MASS * cos_a.powi(2) / total_mass));
        let cart_acc = temp - pole_mass_len * ang_acc * cos_a / total_mass;

        self.position += DT * self.velocity;
        self.velocity += DT * cart_acc;
        self.angle += DT * self.ang_velocity;
        self.ang_velocity += DT * ang_acc;
    }
}

impl Default for CartPole {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment for CartPole {
    fn reset(&mut self) -> State {
        self.position = crate::rand_range(-0.05, 0.05);
        self.velocity = crate::rand_range(-0.05, 0.05);
        self.angle = crate::rand_range(-0.05, 0.05);
        self.ang_velocity = crate::rand_range(-0.05, 0.05);
        self.steps = 0;
        self.make_state()
    }

    fn step(&mut self, action: Action) -> StepResult {
        self.steps += 1;
        let force = if action == 1 { FORCE_MAG } else { -FORCE_MAG };
        self.physics_step(force);

        let done = self.position.abs() > MAX_POS
            || self.angle.abs() > MAX_ANGLE
            || self.steps >= MAX_STEPS;

        StepResult {
            next_state: self.make_state(),
            reward: if done && self.steps < MAX_STEPS {
                0.0
            } else {
                1.0
            },
            done,
            info: if done {
                format!("steps={}", self.steps)
            } else {
                String::new()
            },
        }
    }

    fn action_space(&self) -> usize {
        2
    }
    fn state_space(&self) -> usize {
        4
    }

    fn current_state(&self) -> State {
        self.make_state()
    }
    fn name(&self) -> &str {
        "CartPole-v1"
    }

    fn render(&self) {
        println!(
            "CartPole | pos={:.2} vel={:.2} ang={:.2}° avg={:.2}",
            self.position,
            self.velocity,
            self.angle * 180.0 / PI,
            self.ang_velocity
        );
    }
}
