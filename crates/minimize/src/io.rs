//! Coordinate file I/O utilities for minimization.
//!
//! Reads and writes AMBER inpcrd/rst7 coordinate files with positions kept in
//! Angstroms (the native unit for AMBER force field evaluation).

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use rst_core::amber::inpcrd_writer::{write_inpcrd, InpcrdData};

/// Read an inpcrd file and extract positions and box dimensions.
///
/// Positions and box dimensions are returned in Angstroms (no unit conversion
/// is applied).
///
/// # Arguments
///
/// * `path` - Path to an AMBER inpcrd/rst7 file.
///
/// # Returns
///
/// A tuple of `(positions, box_dimensions)` where `positions` is a `Vec` of
/// `[x, y, z]` coordinate triples and `box_dimensions` is `Some([x, y, z])`
/// when periodic box information is present.
///
/// # Errors
///
/// Returns `Err(String)` if the file cannot be opened or has an invalid format.
#[allow(clippy::type_complexity)]
pub fn read_inpcrd(path: &Path) -> Result<(Vec<[f64; 3]>, Option<[f64; 3]>), String> {
    let file = File::open(path).map_err(|e| format!("Failed to open inpcrd file: {}", e))?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Line 1: Title (skip).
    let _title = lines
        .next()
        .ok_or_else(|| "Empty inpcrd file".to_string())?
        .map_err(|e| format!("Failed to read title line: {}", e))?;

    // Line 2: Atom count (right-justified in first 6 characters).
    let header = lines
        .next()
        .ok_or_else(|| "Missing atom count line in inpcrd".to_string())?
        .map_err(|e| format!("Failed to read atom count line: {}", e))?;

    let n_atoms: usize = header
        .split_whitespace()
        .next()
        .ok_or_else(|| "No atom count found in inpcrd header".to_string())?
        .parse()
        .map_err(|e| format!("Failed to parse atom count: {}", e))?;

    // Lines 3+: Coordinates in 12.7f format, 6 values per line.
    let n_values_needed = n_atoms * 3;
    let mut values: Vec<f64> = Vec::with_capacity(n_values_needed);
    let mut pending_box_line: Option<String> = None;

    for line_result in lines.by_ref() {
        let line = line_result.map_err(|e| format!("Failed to read coordinate line: {}", e))?;

        // Parse fixed-width format (12 characters per value).
        let mut pos = 0;
        while pos + 12 <= line.len() && values.len() < n_values_needed {
            let val_str = line[pos..pos + 12].trim();
            if !val_str.is_empty() {
                let val: f64 = val_str
                    .parse()
                    .map_err(|e| format!("Failed to parse coordinate '{}': {}", val_str, e))?;
                values.push(val);
            }
            pos += 12;
        }

        if values.len() >= n_values_needed {
            // Next line, if present, may be box dimensions.
            if let Some(Ok(box_line)) = lines.next() {
                if !box_line.trim().is_empty() {
                    pending_box_line = Some(box_line);
                }
            }
            break;
        }
    }

    if values.len() < n_values_needed {
        return Err(format!(
            "Expected {} coordinate values, found {}",
            n_values_needed,
            values.len()
        ));
    }

    let positions: Vec<[f64; 3]> = values
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();

    // Parse optional box dimensions (6 values: x, y, z, alpha, beta, gamma).
    let box_dims = if let Some(box_line) = pending_box_line {
        let mut bpos = 0;
        let mut box_vals: Vec<f64> = Vec::with_capacity(6);
        while bpos + 12 <= box_line.len() {
            let val_str = box_line[bpos..bpos + 12].trim();
            if !val_str.is_empty() {
                if let Ok(val) = val_str.parse::<f64>() {
                    box_vals.push(val);
                }
            }
            bpos += 12;
        }
        if box_vals.len() >= 3 {
            Some([box_vals[0], box_vals[1], box_vals[2]])
        } else {
            None
        }
    } else {
        None
    };

    Ok((positions, box_dims))
}

/// Write minimized coordinates to an inpcrd file.
///
/// Uses the `rst_core` inpcrd writer to produce a properly formatted AMBER
/// restart file.  Coordinates and box dimensions must be in Angstroms.
///
/// # Arguments
///
/// * `path` - Destination file path.
/// * `positions` - Atomic coordinates as `[x, y, z]` triples in Angstroms.
/// * `box_dims` - Optional periodic box dimensions `[x, y, z]` in Angstroms.
/// * `title` - Title line for the file header.
///
/// # Errors
///
/// Returns `Err(String)` if the file cannot be created or written.
pub fn write_minimized_coords(
    path: &Path,
    positions: &[[f64; 3]],
    box_dims: Option<[f64; 3]>,
    title: &str,
) -> Result<(), String> {
    let data = InpcrdData {
        title: title.to_string(),
        positions: positions.to_vec(),
        box_dimensions: box_dims,
        box_angles: None, // defaults to [90, 90, 90] in the writer
    };
    write_inpcrd(&data, path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Create a minimal inpcrd file in a temporary directory and return its
    /// path alongside the temp dir handle (so it stays alive).
    fn write_test_inpcrd(
        positions: &[[f64; 3]],
        box_dims: Option<[f64; 3]>,
    ) -> (std::path::PathBuf, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        let path = dir.path().join("test.inpcrd");

        let n_atoms = positions.len();
        let mut content = String::new();
        content.push_str("test title\n");
        content.push_str(&format!("{:>6}\n", n_atoms));

        let mut count = 0;
        for pos in positions {
            for &val in pos {
                content.push_str(&format!("{:12.7}", val));
                count += 1;
                if count % 6 == 0 {
                    content.push('\n');
                }
            }
        }
        if count % 6 != 0 {
            content.push('\n');
        }

        if let Some(dims) = box_dims {
            content.push_str(&format!(
                "{:12.7}{:12.7}{:12.7}{:12.7}{:12.7}{:12.7}\n",
                dims[0], dims[1], dims[2], 90.0, 90.0, 90.0
            ));
        }

        fs::write(&path, &content).expect("failed to write test inpcrd");
        (path, dir)
    }

    #[test]
    fn read_inpcrd_no_box() {
        let positions = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let (path, _dir) = write_test_inpcrd(&positions, None);

        let (coords, box_dims) = read_inpcrd(&path).expect("read_inpcrd failed");
        assert_eq!(coords.len(), 2);
        assert!((coords[0][0] - 1.0).abs() < 1e-6);
        assert!((coords[1][2] - 6.0).abs() < 1e-6);
        assert!(box_dims.is_none());
    }

    #[test]
    fn read_inpcrd_with_box() {
        let positions = vec![[10.0, 20.0, 30.0]];
        let box_dims = Some([30.0, 40.0, 50.0]);
        let (path, _dir) = write_test_inpcrd(&positions, box_dims);

        let (coords, dims) = read_inpcrd(&path).expect("read_inpcrd failed");
        assert_eq!(coords.len(), 1);
        assert!((coords[0][0] - 10.0).abs() < 1e-6);
        let dims = dims.expect("expected box dimensions");
        assert!((dims[0] - 30.0).abs() < 1e-6);
        assert!((dims[1] - 40.0).abs() < 1e-6);
        assert!((dims[2] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn write_and_read_roundtrip() {
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        let path = dir.path().join("roundtrip.inpcrd");

        let positions = vec![[1.1234567, 2.2345678, 3.3456789], [4.0, 5.0, 6.0]];
        let box_dims = Some([50.0, 60.0, 70.0]);

        write_minimized_coords(&path, &positions, box_dims, "roundtrip test")
            .expect("write failed");

        let (coords, dims) = read_inpcrd(&path).expect("read failed");
        assert_eq!(coords.len(), 2);
        assert!((coords[0][0] - 1.1234567).abs() < 1e-6);
        assert!((coords[1][1] - 5.0).abs() < 1e-6);
        let dims = dims.expect("box dimensions missing");
        assert!((dims[0] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn read_inpcrd_empty_file() {
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        let path = dir.path().join("empty.inpcrd");
        fs::write(&path, "").expect("failed to write");
        assert!(read_inpcrd(&path).is_err());
    }

    #[test]
    fn read_inpcrd_missing_file() {
        let path = Path::new("/tmp/nonexistent_inpcrd_file_12345.inpcrd");
        assert!(read_inpcrd(path).is_err());
    }
}
