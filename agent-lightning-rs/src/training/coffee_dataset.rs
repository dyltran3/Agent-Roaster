use crate::core::tensor::Tensor;
/// Specialized dataset loader for Coffee Roasting Profiles.
/// Translates (Temp, Pressure, Time) time-series into Unified MDP Transitions.
use crate::rl::transition::Transition;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub struct CoffeeRoastingProfile {
    pub name: String,
    pub data_points: Vec<RoastingPoint>,
}

pub struct RoastingPoint {
    pub time: f64,
    pub temperature: f64,
    pub pressure: f64,
    pub gas_flow: f64,
    pub drum_speed: f64,
    pub target_temp: f64, // The "optimal" action from the expert profile
}

pub struct CoffeeDataset {
    pub profiles: Vec<CoffeeRoastingProfile>,
}

impl Default for CoffeeDataset {
    fn default() -> Self {
        Self::new()
    }
}

impl CoffeeDataset {
    pub fn new() -> Self {
        CoffeeDataset {
            profiles: Vec::new(),
        }
    }

    /// Load profiles from the raw data directory.
    pub fn load_from_dir<P: AsRef<Path>>(&mut self, dir: P) -> std::io::Result<()> {
        let path = dir.as_ref();
        if !path.is_dir() {
            return Ok(());
        }

        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("csv") {
                if let Ok(profile) = self.parse_csv(&path) {
                    self.profiles.push(profile);
                }
            }
        }
        Ok(())
    }

    fn parse_csv<P: AsRef<Path>>(&self, path: P) -> std::io::Result<CoffeeRoastingProfile> {
        let file = File::open(path.as_ref())?;
        let reader = BufReader::new(file);
        let mut data_points = Vec::new();
        let name = path
            .as_ref()
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();

        for line in reader.lines().skip(1) {
            // Skip header
            let line = line?;
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 6 {
                data_points.push(RoastingPoint {
                    time: parts[0].parse().unwrap_or(0.0),
                    temperature: parts[1].parse().unwrap_or(0.0),
                    pressure: parts[2].parse().unwrap_or(0.0),
                    gas_flow: parts[3].parse().unwrap_or(0.0),
                    drum_speed: parts[4].parse().unwrap_or(0.0),
                    target_temp: parts[5].parse().unwrap_or(0.0),
                });
            }
        }

        Ok(CoffeeRoastingProfile { name, data_points })
    }

    /// Convert roasting profiles into Transitions for LightningRL.
    /// In this context, a "Transition" is a window of state history and a target action.
    pub fn to_transitions(&self) -> Vec<Transition> {
        let mut transitions = Vec::new();
        for profile in &self.profiles {
            for i in 1..profile.data_points.len() {
                let prev = &profile.data_points[i - 1];
                let curr = &profile.data_points[i];

                // State: [time, temp, pressure, gas, speed]
                let state_data = vec![
                    prev.time,
                    prev.temperature,
                    prev.pressure,
                    prev.gas_flow,
                    prev.drum_speed,
                ];
                let input = Tensor::new(state_data.clone(), vec![1, 5]);

                // Action: Target next temperature (simplified as 1D output)
                let output = Tensor::new(vec![curr.target_temp], vec![1, 1]);

                transitions.push(Transition::new(
                    input,
                    output,
                    vec![0.0], // Initial log-prob for offline learning
                    vec![1.0], // Expert data is considered "perfect"
                ));
            }
        }
        transitions
    }
}
