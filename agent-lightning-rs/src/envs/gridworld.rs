/// GridWorld — a 5×5 grid environment for testing RL agents.
///
/// State:  (agent_x, agent_y, goal_x, goal_y) — 4 features
/// Action: 0=Up, 1=Down, 2=Left, 3=Right
/// Reward: +10.0 reaching goal, -0.1 per step, -1.0 hitting wall
use crate::rl::env::{Action, Environment, State, StepResult};

pub struct GridWorld {
    pub width: usize,
    pub height: usize,
    pub agent_x: usize,
    pub agent_y: usize,
    pub goal_x: usize,
    pub goal_y: usize,
    pub max_steps: usize,
    pub current_step: usize,
}

impl GridWorld {
    pub fn new(width: usize, height: usize) -> Self {
        let mut env = GridWorld {
            width,
            height,
            agent_x: 0,
            agent_y: 0,
            goal_x: width - 1,
            goal_y: height - 1,
            max_steps: width * height * 4,
            current_step: 0,
        };
        env.randomize_positions();
        env
    }

    fn randomize_positions(&mut self) {
        self.agent_x = crate::rand_usize(self.width);
        self.agent_y = crate::rand_usize(self.height);
        loop {
            self.goal_x = crate::rand_usize(self.width);
            self.goal_y = crate::rand_usize(self.height);
            if self.goal_x != self.agent_x || self.goal_y != self.agent_y {
                break;
            }
        }
    }

    fn make_state(&self) -> State {
        // Normalize positions to [0, 1]
        vec![
            self.agent_x as f64 / (self.width - 1) as f64,
            self.agent_y as f64 / (self.height - 1) as f64,
            self.goal_x as f64 / (self.width - 1) as f64,
            self.goal_y as f64 / (self.height - 1) as f64,
        ]
    }

    pub fn manhattan_distance(&self) -> usize {
        let dx = (self.agent_x as isize - self.goal_x as isize).unsigned_abs();
        let dy = (self.agent_y as isize - self.goal_y as isize).unsigned_abs();
        dx + dy
    }
}

impl Environment for GridWorld {
    fn reset(&mut self) -> State {
        self.current_step = 0;
        self.randomize_positions();
        self.make_state()
    }

    fn step(&mut self, action: Action) -> StepResult {
        self.current_step += 1;

        let (prev_x, prev_y) = (self.agent_x, self.agent_y);

        // Apply movement
        match action {
            0 => {
                if self.agent_y > 0 {
                    self.agent_y -= 1;
                }
            } // Up
            1 => {
                if self.agent_y + 1 < self.height {
                    self.agent_y += 1;
                }
            } // Down
            2 => {
                if self.agent_x > 0 {
                    self.agent_x -= 1;
                }
            } // Left
            3 => {
                if self.agent_x + 1 < self.width {
                    self.agent_x += 1;
                }
            } // Right
            _ => {}
        }

        let hit_wall = self.agent_x == prev_x && self.agent_y == prev_y;
        let reached_goal = self.agent_x == self.goal_x && self.agent_y == self.goal_y;
        let timeout = self.current_step >= self.max_steps;

        let reward = if reached_goal {
            10.0
        } else if hit_wall {
            -1.0
        } else {
            -0.1
        };

        let done = reached_goal || timeout;

        StepResult {
            next_state: self.make_state(),
            reward,
            done,
            info: if reached_goal {
                "GOAL".to_string()
            } else {
                String::new()
            },
        }
    }

    fn action_space(&self) -> usize {
        4
    }
    fn state_space(&self) -> usize {
        4
    }

    fn current_state(&self) -> State {
        self.make_state()
    }

    fn name(&self) -> &str {
        "GridWorld"
    }

    fn render(&self) {
        for y in 0..self.height {
            for x in 0..self.width {
                if x == self.agent_x && y == self.agent_y {
                    print!("A ");
                } else if x == self.goal_x && y == self.goal_y {
                    print!("G ");
                } else {
                    print!(". ");
                }
            }
            println!();
        }
        println!("Dist: {}", self.manhattan_distance());
    }
}
