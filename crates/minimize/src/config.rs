//! Minimization configuration parameters.
//!
//! Provides [`MinimizeConfig`] for controlling steepest-descent / conjugate-gradient
//! energy minimization runs.  Default values follow AMBER conventions.

/// Configuration for energy minimization.
///
/// All length quantities are in Angstroms and energy quantities in kcal/mol
/// unless otherwise noted.
///
/// # Examples
///
/// ```
/// use rst_minimize::config::MinimizeConfig;
///
/// // Use the defaults (1000 cycles, 100 SD then CG).
/// let cfg = MinimizeConfig::default();
/// assert_eq!(cfg.max_cycles, 1000);
/// assert_eq!(cfg.sd_cycles, 100);
///
/// // Override specific fields.
/// let cfg = MinimizeConfig {
///     max_cycles: 5000,
///     convergence_rms: 0.001,
///     ..MinimizeConfig::default()
/// };
/// assert_eq!(cfg.max_cycles, 5000);
/// ```
#[derive(Debug, Clone)]
pub struct MinimizeConfig {
    /// Maximum number of minimization cycles (default 1000).
    pub max_cycles: usize,
    /// Number of steepest descent cycles before switching to conjugate gradient
    /// (default 100).
    pub sd_cycles: usize,
    /// RMS gradient convergence criterion in kcal/(mol*A) (default 0.01).
    pub convergence_rms: f64,
    /// Non-bonded cutoff distance in Angstroms (default 10.0).
    pub cutoff: f64,
    /// Atom selection mask for positional restraints (`None` = no restraints).
    pub restraint_mask: Option<String>,
    /// Force constant for positional restraints in kcal/(mol*A^2) (default 0.0).
    pub restraint_weight: f64,
    /// Initial step size for line minimization (default 0.01).
    pub initial_step_size: f64,
}

impl Default for MinimizeConfig {
    fn default() -> Self {
        Self {
            max_cycles: 1000,
            sd_cycles: 100,
            convergence_rms: 0.01,
            cutoff: 10.0,
            restraint_mask: None,
            restraint_weight: 0.0,
            initial_step_size: 0.01,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_values() {
        let cfg = MinimizeConfig::default();
        assert_eq!(cfg.max_cycles, 1000);
        assert_eq!(cfg.sd_cycles, 100);
        assert!((cfg.convergence_rms - 0.01).abs() < f64::EPSILON);
        assert!((cfg.cutoff - 10.0).abs() < f64::EPSILON);
        assert!(cfg.restraint_mask.is_none());
        assert!((cfg.restraint_weight - 0.0).abs() < f64::EPSILON);
        assert!((cfg.initial_step_size - 0.01).abs() < f64::EPSILON);
    }

    #[test]
    fn override_fields() {
        let cfg = MinimizeConfig {
            max_cycles: 5000,
            sd_cycles: 500,
            convergence_rms: 0.001,
            cutoff: 12.0,
            restraint_mask: Some(":WAT".to_string()),
            restraint_weight: 5.0,
            initial_step_size: 0.005,
        };
        assert_eq!(cfg.max_cycles, 5000);
        assert_eq!(cfg.sd_cycles, 500);
        assert!((cfg.convergence_rms - 0.001).abs() < f64::EPSILON);
        assert!((cfg.cutoff - 12.0).abs() < f64::EPSILON);
        assert_eq!(cfg.restraint_mask.as_deref(), Some(":WAT"));
        assert!((cfg.restraint_weight - 5.0).abs() < f64::EPSILON);
        assert!((cfg.initial_step_size - 0.005).abs() < f64::EPSILON);
    }

    #[test]
    fn clone_and_debug() {
        let cfg = MinimizeConfig::default();
        let cfg2 = cfg.clone();
        assert_eq!(cfg2.max_cycles, cfg.max_cycles);
        // Debug should not panic.
        let _ = format!("{:?}", cfg);
    }
}
