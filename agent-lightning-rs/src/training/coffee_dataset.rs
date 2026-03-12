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
    pub pressure: f64,    // Normalized or secondary temp
    pub gas_flow: f64,    // gas_percent
    pub drum_speed: f64,  // drum_rpm
    pub target_temp: f64, // the expert action (next temperature or gas)
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
            let ext = path.extension().and_then(|s| s.to_str());
            if ext == Some("csv") {
                if let Ok(profile) = self.parse_csv(&path) {
                    self.profiles.push(profile);
                }
            } else if ext == Some("json") {
                if let Ok(profile) = self.parse_json(&path) {
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

        let mut lines = reader.lines();
        let header = match lines.next() {
            Some(Ok(h)) => h,
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Empty CSV",
                ))
            }
        };

        let cols: Vec<&str> = header.split(',').collect();
        let find_idx = |name: &str| cols.iter().position(|&c| c == name);

        // Detect if it's the new style or old style
        let time_idx = find_idx("time_sec").or(find_idx("time")).unwrap_or(0);
        let bt_idx = find_idx("bt_c").or(find_idx("temperature")).unwrap_or(1);
        let et_idx = find_idx("et_c").or(find_idx("pressure")).unwrap_or(2);
        let gas_idx = find_idx("gas_percent")
            .or(find_idx("gas_flow"))
            .unwrap_or(3);
        let rpm_idx = find_idx("drum_rpm").or(find_idx("drum_speed")).unwrap_or(4);
        let target_idx = find_idx("ror_bt").or(find_idx("target_temp")).unwrap_or(5);

        for line in lines {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() > target_idx {
                data_points.push(RoastingPoint {
                    time: parts[time_idx].parse().unwrap_or(0.0),
                    temperature: parts[bt_idx].parse().unwrap_or(0.0),
                    pressure: parts[et_idx].parse().unwrap_or(0.0),
                    gas_flow: parts[gas_idx].parse().unwrap_or(0.0),
                    drum_speed: parts[rpm_idx].parse().unwrap_or(0.0),
                    target_temp: parts[target_idx].parse().unwrap_or(0.0),
                });
            }
        }

        Ok(CoffeeRoastingProfile { name, data_points })
    }

    /// Manual JSON parsing for roasting profiles (No Serde dependency)
    fn parse_json<P: AsRef<Path>>(&self, path: P) -> std::io::Result<CoffeeRoastingProfile> {
        let content = std::fs::read_to_string(path.as_ref())?;
        let mut data_points = Vec::new();
        let name = path
            .as_ref()
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();

        // Very basic manual extraction for "timeseries" array in the demo JSON
        // Since we don't have Serde, we look for objects inside the timeseries block
        if let Some(start_idx) = content.find("\"timeseries\": [") {
            let timeseries_block = &content[start_idx..];
            let mut current = timeseries_block;

            while let Some(obj_start) = current.find("{") {
                let obj_end = current.find("}").unwrap_or(current.len());
                let obj = &current[obj_start..obj_end];

                let get_val = |key: &str| -> f64 {
                    let key_str = format!("\"{}\":", key);
                    if let Some(k_idx) = obj.find(&key_str) {
                        let val_part = &obj[k_idx + key_str.len()..];
                        let val_str = val_part
                            .split(|c| c == ',' || c == '}' || c == ' ' || c == '\n')
                            .next()
                            .unwrap_or("0");
                        val_str.trim().parse().unwrap_or(0.0)
                    } else {
                        0.0
                    }
                };

                data_points.push(RoastingPoint {
                    time: get_val("time_sec"),
                    temperature: get_val("bt_c"),
                    pressure: get_val("et_c"),
                    gas_flow: get_val("gas_percent"),
                    drum_speed: get_val("drum_rpm"),
                    target_temp: get_val("ror_bt"),
                });

                current = &current[obj_end + 1..];
                if let Some(list_end) = current.find("]") {
                    if list_end < current.find("{").unwrap_or(current.len() + 1) {
                        break;
                    }
                }
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
