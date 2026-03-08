/// **Extended Kalman Filter (EKF)** for Agent-Roaster (Physics-Informed Edge AI)
/// Uses Thermodynamic equations (Coffee Master Reference F-21 to F-25)
/// to estimate hidden state variables from noisy sensors.
/// State X = [T_bean, ROR, Moisture (X), CDI (Color Development Index)]
pub struct ExtendedKalmanFilter {
    pub x: [f64; 4],      // State estimate
    pub p: [[f64; 4]; 4], // Covariance matrix
    pub q: [[f64; 4]; 4], // Process noise covariance
    pub r: [[f64; 2]; 2], // Measurement noise covariance (ET, BT)
}

impl ExtendedKalmanFilter {
    pub fn new(initial_bt: f64) -> Self {
        ExtendedKalmanFilter {
            x: [initial_bt, 0.0, 11.0, 0.0], // Init T_bean, ROR=0, Moisture=11%, CDI=0
            p: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            q: [
                [0.1, 0.0, 0.0, 0.0],
                [0.0, 0.1, 0.0, 0.0],
                [0.0, 0.0, 0.001, 0.0],
                [0.0, 0.0, 0.0, 0.01],
            ],
            r: [
                [2.0, 0.0], // ET sensor noise
                [0.0, 0.5], // BT sensor noise
            ],
        }
    }

    /// Extrapolate the state to the next time step based on Physics.
    /// Uses F-21 standard notation for State Prediction.
    pub fn predict(&mut self, dt: f64, et: f64) {
        let t_bean = self.x[0];
        let ror = self.x[1];
        let moisture = self.x[2];
        let cdi = self.x[3];

        // Non-linear thermodynamic predictions (F-21 abstraction)
        // 1. Newton's Law of Cooling: dT/dt = k * (ET - BT) -> ROR
        let k_heat_transfer = 0.02; // Thermodynamic abstract constant
        let predicted_ror = ror + 0.1 * (k_heat_transfer * (et - t_bean) - ror);
        let predicted_t_bean = t_bean + predicted_ror * dt;

        // 2. Moisture loss (evaporation accelerates > 100°C)
        let evap_rate = if t_bean > 100.0 {
            0.05 * (t_bean - 100.0) / 100.0
        } else {
            0.001
        };
        let predicted_moisture = f64::max(0.0, moisture - evap_rate * dt);

        // 3. Maillard Reaction & Color (CDI starts accumulating quickly > 150°C)
        let color_rate = if t_bean > 150.0 {
            0.01 * (t_bean - 150.0)
        } else {
            0.0
        };
        let predicted_cdi = cdi + color_rate * dt;

        self.x = [
            predicted_t_bean,
            predicted_ror,
            predicted_moisture,
            predicted_cdi,
        ];

        // Linearized Covariance prediction: P = F * P * F^T + Q
        // (Simplified diagonal propagation for Edge AI computational limits)
        for i in 0..4 {
            self.p[i][i] = self.p[i][i] * 1.01 + self.q[i][i];
        }
    }

    /// Measurement update combining Sensor data with Prediction
    /// Equations F-22 to F-25
    pub fn update(&mut self, sensor_bt: f64) {
        // Measurement model: z = H*x
        // We observe BT directly from the noisy sensor.
        let z = sensor_bt;
        let h_x = self.x[0]; // predicted BT measurement

        // Innovation (Residual F-23)
        let y = z - h_x;

        // Innovation covariance (F-24)
        let h_p_h = self.p[0][0];
        let s = h_p_h + self.r[1][1];

        // Kalman Gain (F-25)
        let k_gain = self.p[0][0] / s;

        // Update State
        self.x[0] += k_gain * y;

        // ROR is highly correlated with BT, we update ROR as well via cross-correlation assumption
        let k_gain_ror = self.p[1][1] / (self.p[1][1] + s);
        self.x[1] += (k_gain_ror * 0.1) * y; // distribute innovation to derivative

        // Update Covariance P = (I - K*H) * P
        self.p[0][0] = (1.0 - k_gain) * self.p[0][0];
    }
}
