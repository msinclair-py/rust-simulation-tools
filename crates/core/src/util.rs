//! Common utility functions shared across modules.

/// Compute squared distance between two 3D points.
#[inline(always)]
pub fn distance_squared(p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
    let dx = p1[0] - p2[0];
    let dy = p1[1] - p2[1];
    let dz = p1[2] - p2[2];
    dx * dx + dy * dy + dz * dz
}

/// Compute squared distance from components.
#[inline(always)]
pub fn distance_squared_components(dx: f64, dy: f64, dz: f64) -> f64 {
    dx * dx + dy * dy + dz * dz
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_squared() {
        let p1 = [0.0, 0.0, 0.0];
        let p2 = [1.0, 0.0, 0.0];
        assert!((distance_squared(&p1, &p2) - 1.0).abs() < 1e-10);

        let p3 = [1.0, 1.0, 1.0];
        assert!((distance_squared(&p1, &p3) - 3.0).abs() < 1e-10);
    }
}
