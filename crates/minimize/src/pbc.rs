//! Periodic boundary conditions for rectangular boxes.
//!
//! Implements the minimum-image convention used by AMBER-style molecular
//! simulations with orthorhombic periodic boxes.

/// Rectangular periodic box with cached half-dimensions.
///
/// All lengths are in Angstroms.
///
/// # Examples
///
/// ```
/// use rst_minimize::pbc::PeriodicBox;
///
/// let pbox = PeriodicBox::new([30.0, 30.0, 30.0]);
///
/// // A displacement of 20 A wraps to -10 A via minimum image.
/// let mut dx = 20.0;
/// let mut dy = 0.0;
/// let mut dz = 0.0;
/// pbox.minimum_image(&mut dx, &mut dy, &mut dz);
/// assert!((dx - (-10.0)).abs() < 1e-10);
/// ```
#[derive(Debug, Clone)]
pub struct PeriodicBox {
    /// Full box edge lengths [x, y, z] in Angstroms.
    pub dimensions: [f64; 3],
    /// Half box edge lengths, cached for minimum-image checks.
    pub half_dimensions: [f64; 3],
}

impl PeriodicBox {
    /// Create a new rectangular periodic box from its edge lengths.
    ///
    /// # Arguments
    ///
    /// * `dimensions` - Box edge lengths `[x, y, z]` in Angstroms.
    pub fn new(dimensions: [f64; 3]) -> Self {
        let half_dimensions = [
            dimensions[0] * 0.5,
            dimensions[1] * 0.5,
            dimensions[2] * 0.5,
        ];
        Self {
            dimensions,
            half_dimensions,
        }
    }

    /// Apply minimum image convention to a displacement vector.
    ///
    /// Modifies `dx`, `dy`, `dz` in place so that each component lies in the
    /// range `(-L/2, L/2]` where `L` is the corresponding box edge length.
    ///
    /// Uses the standard `floor(d / L + 0.5) * L` formula.
    #[inline]
    pub fn minimum_image(&self, dx: &mut f64, dy: &mut f64, dz: &mut f64) {
        *dx -= (*dx / self.dimensions[0] + 0.5).floor() * self.dimensions[0];
        *dy -= (*dy / self.dimensions[1] + 0.5).floor() * self.dimensions[1];
        *dz -= (*dz / self.dimensions[2] + 0.5).floor() * self.dimensions[2];
    }

    /// Apply minimum image convention and return the squared distance between
    /// two positions.
    ///
    /// This is equivalent to computing the displacement, applying
    /// [`minimum_image`](Self::minimum_image), and then returning the squared
    /// norm, but avoids creating a temporary displacement variable.
    #[inline]
    pub fn distance_squared(&self, pos1: &[f64; 3], pos2: &[f64; 3]) -> f64 {
        let mut dx = pos1[0] - pos2[0];
        let mut dy = pos1[1] - pos2[1];
        let mut dz = pos1[2] - pos2[2];
        self.minimum_image(&mut dx, &mut dy, &mut dz);
        dx * dx + dy * dy + dz * dz
    }

    /// Wrap a position into the primary box `[0, L)` along each axis.
    ///
    /// Modifies the position array in place.
    #[inline]
    pub fn wrap_position(&self, pos: &mut [f64; 3]) {
        for i in 0..3 {
            pos[i] -= (pos[i] / self.dimensions[i]).floor() * self.dimensions[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < TOL
    }

    #[test]
    fn new_computes_half_dimensions() {
        let pbox = PeriodicBox::new([30.0, 40.0, 50.0]);
        assert!(approx_eq(pbox.half_dimensions[0], 15.0));
        assert!(approx_eq(pbox.half_dimensions[1], 20.0));
        assert!(approx_eq(pbox.half_dimensions[2], 25.0));
    }

    #[test]
    fn minimum_image_no_wrap() {
        let pbox = PeriodicBox::new([30.0, 30.0, 30.0]);
        let mut dx = 5.0;
        let mut dy = -5.0;
        let mut dz = 0.0;
        pbox.minimum_image(&mut dx, &mut dy, &mut dz);
        assert!(approx_eq(dx, 5.0));
        assert!(approx_eq(dy, -5.0));
        assert!(approx_eq(dz, 0.0));
    }

    #[test]
    fn minimum_image_wraps_positive() {
        let pbox = PeriodicBox::new([30.0, 30.0, 30.0]);
        let mut dx = 20.0;
        let mut dy = 25.0;
        let mut dz = 29.0;
        pbox.minimum_image(&mut dx, &mut dy, &mut dz);
        assert!(approx_eq(dx, -10.0));
        assert!(approx_eq(dy, -5.0));
        assert!(approx_eq(dz, -1.0));
    }

    #[test]
    fn minimum_image_wraps_negative() {
        let pbox = PeriodicBox::new([30.0, 30.0, 30.0]);
        let mut dx = -20.0;
        let mut dy = -25.0;
        let mut dz = -29.0;
        pbox.minimum_image(&mut dx, &mut dy, &mut dz);
        assert!(approx_eq(dx, 10.0));
        assert!(approx_eq(dy, 5.0));
        assert!(approx_eq(dz, 1.0));
    }

    #[test]
    fn distance_squared_across_boundary() {
        let pbox = PeriodicBox::new([10.0, 10.0, 10.0]);
        let pos1 = [1.0, 1.0, 1.0];
        let pos2 = [9.0, 9.0, 9.0];
        // Wrapped displacements: 2.0, 2.0, 2.0  => dist^2 = 12.0
        let d2 = pbox.distance_squared(&pos1, &pos2);
        assert!(approx_eq(d2, 12.0));
    }

    #[test]
    fn distance_squared_no_wrap() {
        let pbox = PeriodicBox::new([100.0, 100.0, 100.0]);
        let pos1 = [10.0, 20.0, 30.0];
        let pos2 = [13.0, 24.0, 30.0];
        let d2 = pbox.distance_squared(&pos1, &pos2);
        // 9 + 16 + 0 = 25
        assert!(approx_eq(d2, 25.0));
    }

    #[test]
    fn wrap_position_positive_overflow() {
        let pbox = PeriodicBox::new([10.0, 10.0, 10.0]);
        let mut pos = [12.0, 25.0, -3.0];
        pbox.wrap_position(&mut pos);
        assert!(approx_eq(pos[0], 2.0));
        assert!(approx_eq(pos[1], 5.0));
        assert!(approx_eq(pos[2], 7.0));
    }

    #[test]
    fn wrap_position_already_inside() {
        let pbox = PeriodicBox::new([10.0, 10.0, 10.0]);
        let mut pos = [5.0, 5.0, 5.0];
        pbox.wrap_position(&mut pos);
        assert!(approx_eq(pos[0], 5.0));
        assert!(approx_eq(pos[1], 5.0));
        assert!(approx_eq(pos[2], 5.0));
    }

    #[test]
    fn clone_and_debug() {
        let pbox = PeriodicBox::new([10.0, 20.0, 30.0]);
        let pbox2 = pbox.clone();
        assert!(approx_eq(pbox.dimensions[0], pbox2.dimensions[0]));
        let _ = format!("{:?}", pbox);
    }
}
