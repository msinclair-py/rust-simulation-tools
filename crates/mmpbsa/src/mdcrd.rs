//! AMBER ASCII trajectory (mdcrd/crd) reader.
//!
//! Reads Cpptraj-generated `.mdcrd` files frame by frame.
//! Format: title line, then coordinate values in %8.3f format (10 per line),
//! with n_atoms*3 values per frame. Coordinates are in Angstroms.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Streaming reader for AMBER ASCII trajectory files.
pub struct MdcrdReader {
    reader: BufReader<File>,
    n_atoms: usize,
    has_box: bool,
    current_frame: usize,
}

impl MdcrdReader {
    /// Open an mdcrd file for reading.
    ///
    /// # Arguments
    /// * `path` - Path to the mdcrd file
    /// * `n_atoms` - Number of atoms (must be known from the topology)
    /// * `has_box` - Whether the trajectory contains box dimensions
    pub fn open<P: AsRef<Path>>(path: P, n_atoms: usize, has_box: bool) -> Result<Self, String> {
        let file =
            File::open(path.as_ref()).map_err(|e| format!("Failed to open mdcrd: {}", e))?;
        let mut reader = BufReader::new(file);

        // Skip title line
        let mut title = String::new();
        reader
            .read_line(&mut title)
            .map_err(|e| format!("Failed to read mdcrd title: {}", e))?;

        Ok(Self {
            reader,
            n_atoms,
            has_box,
            current_frame: 0,
        })
    }

    /// Read the next frame. Returns None at EOF.
    ///
    /// Coordinates are returned in Angstroms as `Vec<[f64; 3]>`.
    pub fn read_frame(&mut self) -> Result<Option<Vec<[f64; 3]>>, String> {
        let n_values = self.n_atoms * 3 + if self.has_box { 6 } else { 0 };
        let mut values = Vec::with_capacity(n_values);
        let mut line = String::new();

        while values.len() < n_values {
            line.clear();
            let bytes = self
                .reader
                .read_line(&mut line)
                .map_err(|e| format!("Failed to read mdcrd line: {}", e))?;
            if bytes == 0 {
                // EOF
                if values.is_empty() {
                    return Ok(None);
                } else {
                    return Err(format!(
                        "Unexpected EOF at frame {} (got {}/{} values)",
                        self.current_frame,
                        values.len(),
                        n_values
                    ));
                }
            }

            // Parse fixed-width 8-character fields
            let trimmed = line.trim_end_matches('\n').trim_end_matches('\r');
            let mut pos = 0;
            while pos < trimmed.len() && values.len() < n_values {
                let end = (pos + 8).min(trimmed.len());
                let field = &trimmed[pos..end];
                let val: f64 = field.trim().parse().map_err(|e| {
                    format!(
                        "Failed to parse mdcrd value '{}' at frame {}: {}",
                        field.trim(),
                        self.current_frame,
                        e
                    )
                })?;
                values.push(val);
                pos = end;
            }
        }

        // Convert flat values to [f64; 3] coordinates (skip box if present)
        let mut coords = Vec::with_capacity(self.n_atoms);
        for i in 0..self.n_atoms {
            coords.push([
                values[i * 3],
                values[i * 3 + 1],
                values[i * 3 + 2],
            ]);
        }

        self.current_frame += 1;
        Ok(Some(coords))
    }

    pub fn current_frame(&self) -> usize {
        self.current_frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_mdcrd() {
        let path = "/Users/msinclair/testing_grounds/mmpbsa/tutorial_files/_MMPBSA_complex.mdcrd.0";
        if !std::path::Path::new(path).exists() {
            return;
        }

        // Get n_atoms from the complex prmtop
        let top = rst_core::amber::prmtop::parse_prmtop(
            "/Users/msinclair/testing_grounds/mmpbsa/tutorial_files/ras-raf.prmtop",
        )
        .expect("Failed to parse prmtop");

        let mut reader = MdcrdReader::open(path, top.n_atoms, false).expect("Failed to open mdcrd");

        // Read first frame
        let frame = reader.read_frame().expect("Failed to read frame");
        assert!(frame.is_some(), "Should have at least one frame");
        let coords = frame.unwrap();
        assert_eq!(coords.len(), top.n_atoms);

        // First atom coords should match the file header
        assert!((coords[0][0] - 41.300).abs() < 0.001);
        assert!((coords[0][1] - 10.180).abs() < 0.001);
        assert!((coords[0][2] - 47.356).abs() < 0.001);
    }
}
