/// **Agent-Roaster Safety Bounds System**
/// Hard-coded physical rules and fallback mechanisms to prevent hardware damage.
/// Implements thermodynamics constraints F-19 & F-30.

/// Computes the Base Feedforward Setpoint (F-19 abstraction).
/// Thermodynamic physics: Base Gas is required to maintain T context and push target ROR.
pub fn compute_base_gas(et: f64, bt: f64, target_ror: f64) -> f64 {
    let k_h = 0.5; // Heat transfer coefficient
    let k_r = 2.0; // ROR acceleration coefficient

    let base = k_h * (1.1 * et - bt).max(0.0) + k_r * target_ror;
    base.clamp(0.0, 100.0)
}

/// Applies AI's Residual Correction (F-30 abstraction) over the Base Gas.
/// The RL Agent is NOT allowed to directly output [0, 100].
/// It outputs a normalized Tanh residual [-1, 1], which corresponds to +/- 5%.
pub fn apply_hybrid_control(base_gas: f64, ai_residual_output: f64) -> f64 {
    let max_correction = 5.0; // ±5% authority is given to AI
    let action = base_gas + ai_residual_output * max_correction;

    // Strict bounding
    action.clamp(0.0, 100.0)
}

/// System-level safety bounds monitor.
/// Returns true if the system is safe, false if an Emergency Stop (e-stop) is needed.
pub fn check_safety_bounds(et: f64, bt: f64, ror: f64) -> bool {
    // 1. Drum overheating critical limit
    if et > 250.0 {
        return false;
    }
    // 2. BT sudden crash during high heat (potential sensor failure or fire)
    if bt > 180.0 && ror < -10.0 {
        return false;
    }
    // 3. Thermal Runaway (ROR accelerating uncontrollably)
    if ror > 30.0 {
        return false;
    }

    true
}
