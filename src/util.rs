//! Common utility functions shared across modules
//!
//! This module provides reusable utilities for:
//! - Converting Python integer arrays to Vec<usize>
//! - Common geometric operations

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// Convert any integer numpy array to Vec<usize>
///
/// Supports i32, i64, u64, and usize array types.
/// Optionally validates that values are non-negative.
///
/// # Arguments
/// * `array` - A bound PyAny reference to a numpy array
/// * `name` - Name of the parameter for error messages
/// * `allow_negative` - If false, returns error for negative values
///
/// # Returns
/// * `Ok(Vec<usize>)` - Successfully converted indices
/// * `Err(PyErr)` - Type error or validation error
#[allow(dead_code)]
pub fn extract_indices<'py>(
    array: &Bound<'py, PyAny>,
    name: &str,
    allow_negative: bool,
) -> PyResult<Vec<usize>> {
    // Try i64 first (most common from numpy)
    if let Ok(arr) = array.extract::<PyReadonlyArray1<i64>>() {
        return arr
            .as_array()
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                if !allow_negative && x < 0 {
                    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "{} index {} cannot be negative: {}",
                        name, i, x
                    )))
                } else {
                    Ok(x as usize)
                }
            })
            .collect();
    }

    // Try i32
    if let Ok(arr) = array.extract::<PyReadonlyArray1<i32>>() {
        return arr
            .as_array()
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                if !allow_negative && x < 0 {
                    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "{} index {} cannot be negative: {}",
                        name, i, x
                    )))
                } else {
                    Ok(x as usize)
                }
            })
            .collect();
    }

    // Try u64
    if let Ok(arr) = array.extract::<PyReadonlyArray1<u64>>() {
        return Ok(arr.as_array().iter().map(|&x| x as usize).collect());
    }

    // Try usize
    if let Ok(arr) = array.extract::<PyReadonlyArray1<usize>>() {
        return Ok(arr.as_array().to_vec());
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
        "{} must be an integer array",
        name
    )))
}

/// Extract indices without negative value checking (faster path)
#[allow(dead_code)]
pub fn extract_indices_unchecked<'py>(
    array: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Vec<usize>> {
    // Try i64 first (most common from numpy)
    if let Ok(arr) = array.extract::<PyReadonlyArray1<i64>>() {
        return Ok(arr.as_array().iter().map(|&x| x as usize).collect());
    }

    // Try i32
    if let Ok(arr) = array.extract::<PyReadonlyArray1<i32>>() {
        return Ok(arr.as_array().iter().map(|&x| x as usize).collect());
    }

    // Try u64
    if let Ok(arr) = array.extract::<PyReadonlyArray1<u64>>() {
        return Ok(arr.as_array().iter().map(|&x| x as usize).collect());
    }

    // Try usize
    if let Ok(arr) = array.extract::<PyReadonlyArray1<usize>>() {
        return Ok(arr.as_array().to_vec());
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
        "{} must be an integer array",
        name
    )))
}

/// Compute squared distance between two 3D points
#[inline(always)]
pub fn distance_squared(p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
    let dx = p1[0] - p2[0];
    let dy = p1[1] - p2[1];
    let dz = p1[2] - p2[2];
    dx * dx + dy * dy + dz * dz
}

/// Compute squared distance from components
#[inline(always)]
#[allow(dead_code)]
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
