use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// A utility to export training metrics to a CSV file.
pub struct TrainingExporter {
    export_dir: String,
    filename: String,
    records: Vec<TrainingRecord>,
}

pub struct TrainingRecord {
    pub episode: u64,
    pub reward: f64,
    pub loss: f64,
}

impl TrainingExporter {
    pub fn new(prefix: &str) -> Self {
        let export_dir = "data/exports".to_string();

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let filename = format!("{}_{}.csv", prefix, timestamp);

        Self {
            export_dir,
            filename,
            records: Vec::new(),
        }
    }

    /// Add a new record to the exporter.
    pub fn add_record(&mut self, episode: u64, reward: f64, loss: f64) {
        self.records.push(TrainingRecord {
            episode,
            reward,
            loss,
        });
    }

    /// Export the collected records to a CSV file.
    pub fn export(&self) -> std::io::Result<()> {
        let dir_path = Path::new(&self.export_dir);
        if !dir_path.exists() {
            fs::create_dir_all(dir_path)?;
        }

        let file_path = dir_path.join(&self.filename);
        let mut file = File::create(&file_path)?;

        // Write header
        writeln!(file, "episode,reward,loss")?;

        // Write records
        for record in &self.records {
            writeln!(
                file,
                "{},{:.4},{:.4}",
                record.episode, record.reward, record.loss
            )?;
        }

        println!(
            "\n  [EXPORTER] ✅ Successfully exported {} records to: {}",
            self.records.len(),
            file_path.display()
        );

        Ok(())
    }
}
