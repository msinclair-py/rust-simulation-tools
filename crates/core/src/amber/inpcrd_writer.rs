//! AMBER inpcrd/rst7 coordinate file writer.
//!
//! Writes AMBER coordinate/restart files from atomic positions.
//! Coordinates are expected in Angstroms (no unit conversion is performed).

use std::fmt::Write as FmtWrite;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Maximum length for the title line in an inpcrd file.
const MAX_TITLE_LENGTH: usize = 80;

/// Number of coordinate values per line in the inpcrd format.
const VALUES_PER_LINE: usize = 6;

/// Data needed to write an inpcrd file.
#[derive(Debug, Clone)]
pub struct InpcrdData {
    /// Title line (truncated to 80 characters on write).
    pub title: String,
    /// Coordinates in Angstroms, one [x, y, z] per atom.
    pub positions: Vec<[f64; 3]>,
    /// Box dimensions [x, y, z] in Angstroms (optional).
    pub box_dimensions: Option<[f64; 3]>,
    /// Box angles [alpha, beta, gamma] in degrees (defaults to [90, 90, 90] if
    /// `box_dimensions` is `Some` but this field is `None`).
    pub box_angles: Option<[f64; 3]>,
}

/// Write an inpcrd file to disk.
///
/// # Arguments
/// * `data` - The coordinate data to write.
/// * `path` - Destination file path.
///
/// # Errors
/// Returns `Err(String)` if the file cannot be created or written.
pub fn write_inpcrd(data: &InpcrdData, path: &Path) -> Result<(), String> {
    let content = write_inpcrd_string(data)?;
    let mut file =
        File::create(path).map_err(|e| format!("Failed to create inpcrd file: {}", e))?;
    file.write_all(content.as_bytes())
        .map_err(|e| format!("Failed to write inpcrd file: {}", e))?;
    Ok(())
}

/// Write an inpcrd to a `String`.
///
/// # Arguments
/// * `data` - The coordinate data to write.
///
/// # Errors
/// Returns `Err(String)` if formatting fails (should not happen under normal
/// circumstances).
pub fn write_inpcrd_string(data: &InpcrdData) -> Result<String, String> {
    let n_atoms = data.positions.len();

    // Pre-allocate: title + header + coords + optional box.
    // Each coord line is ~72 chars; rough estimate avoids repeated reallocation.
    let coord_lines = (n_atoms * 3).div_ceil(VALUES_PER_LINE);
    let estimated_capacity = MAX_TITLE_LENGTH + 10 + coord_lines * 74 + 80;
    let mut output = String::with_capacity(estimated_capacity);

    // Line 1: Title (truncated to 80 characters).
    let title = if data.title.len() > MAX_TITLE_LENGTH {
        &data.title[..MAX_TITLE_LENGTH]
    } else {
        &data.title
    };
    writeln!(output, "{}", title).map_err(|e| format!("Format error: {}", e))?;

    // Line 2: Atom count, right-justified in 6 characters.
    writeln!(output, "{:>6}", n_atoms).map_err(|e| format!("Format error: {}", e))?;

    // Lines 3+: Coordinates in Fortran (6F12.7) format.
    write_coordinates(&mut output, &data.positions)?;

    // Optional box line.
    if let Some(dims) = &data.box_dimensions {
        let angles = data.box_angles.unwrap_or([90.0, 90.0, 90.0]);
        writeln!(
            output,
            "{:12.7}{:12.7}{:12.7}{:12.7}{:12.7}{:12.7}",
            dims[0], dims[1], dims[2], angles[0], angles[1], angles[2],
        )
        .map_err(|e| format!("Format error: {}", e))?;
    }

    Ok(output)
}

/// Write coordinate values in Fortran `(6F12.7)` fixed-width format.
///
/// Flattens the `[x, y, z]` triples and writes 6 values per line, each
/// formatted as 12 characters wide with 7 decimal places.
fn write_coordinates(output: &mut String, positions: &[[f64; 3]]) -> Result<(), String> {
    let mut count = 0;
    for pos in positions {
        for &val in pos {
            write!(output, "{:12.7}", val).map_err(|e| format!("Format error: {}", e))?;
            count += 1;
            if count % VALUES_PER_LINE == 0 {
                output.push('\n');
            }
        }
    }
    // If the last line was not complete, terminate it.
    if count % VALUES_PER_LINE != 0 {
        output.push('\n');
    }
    Ok(())
}
