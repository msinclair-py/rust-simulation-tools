//! DCD trajectory file reader.
//!
//! Reads CHARMM/NAMD/X-PLOR DCD binary trajectory format.
//! Coordinates are converted from Angstrom to nm.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// Angstrom to nm conversion factor
pub const ANGSTROM_TO_NM: f64 = 0.1;

// ============================================================================
// Data Structures
// ============================================================================

/// DCD file header information.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct DcdHeader {
    /// Number of frames in trajectory
    pub n_frames: usize,
    /// Number of atoms
    pub n_atoms: usize,
    /// Starting timestep
    pub start_timestep: i32,
    /// Timestep interval between frames
    pub timestep_interval: i32,
    /// Integration timestep in AKMA units
    pub timestep: f32,
    /// Whether unit cell information is present
    pub has_unit_cell: bool,
    /// Whether this is a CHARMM format DCD
    pub is_charmm: bool,
    /// Whether coordinates are stored as 64-bit (CHARMM extended)
    pub is_64bit: bool,
    /// Whether the file is big-endian
    pub is_big_endian: bool,
    /// Title strings from header
    pub titles: Vec<String>,
    /// File byte offset where frame data begins
    pub first_frame_offset: u64,
    /// Size in bytes of each frame (for seeking)
    pub frame_size: usize,
}

/// DCD trajectory reader with streaming capability.
pub struct DcdReader {
    reader: BufReader<File>,
    header: DcdHeader,
    current_frame: usize,
}

impl DcdReader {
    /// Open a DCD file for reading.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let file =
            File::open(path.as_ref()).map_err(|e| format!("Failed to open DCD file: {}", e))?;
        let mut reader = BufReader::new(file);
        let header = read_dcd_header_internal(&mut reader)?;

        Ok(Self {
            reader,
            header,
            current_frame: 0,
        })
    }

    /// Get the header information.
    pub fn header(&self) -> &DcdHeader {
        &self.header
    }

    /// Get the number of frames.
    pub fn n_frames(&self) -> usize {
        self.header.n_frames
    }

    /// Get the number of atoms.
    pub fn n_atoms(&self) -> usize {
        self.header.n_atoms
    }

    /// Read the next frame, returning positions in nm.
    /// Returns None if at end of trajectory.
    pub fn read_frame(&mut self) -> Result<Option<(Vec<[f64; 3]>, Option<[f64; 6]>)>, String> {
        if self.current_frame >= self.header.n_frames {
            return Ok(None);
        }

        let result = read_frame_internal(&mut self.reader, &self.header)?;
        self.current_frame += 1;
        Ok(Some(result))
    }

    /// Seek to a specific frame.
    pub fn seek_frame(&mut self, frame: usize) -> Result<(), String> {
        if frame >= self.header.n_frames {
            return Err(format!(
                "Frame {} out of range (0-{})",
                frame,
                self.header.n_frames - 1
            ));
        }

        let offset =
            self.header.first_frame_offset + (frame as u64) * (self.header.frame_size as u64);
        self.reader
            .seek(SeekFrom::Start(offset))
            .map_err(|e| format!("Seek failed: {}", e))?;
        self.current_frame = frame;
        Ok(())
    }

    /// Read all frames into memory.
    /// Returns (positions, box_info) where positions has shape (n_frames, n_atoms, 3).
    pub fn read_all_frames(
        &mut self,
    ) -> Result<(Vec<Vec<[f64; 3]>>, Vec<Option<[f64; 6]>>), String> {
        self.seek_frame(0)?;

        let mut all_positions = Vec::with_capacity(self.header.n_frames);
        let mut all_boxes = Vec::with_capacity(self.header.n_frames);

        while let Some((positions, box_info)) = self.read_frame()? {
            all_positions.push(positions);
            all_boxes.push(box_info);
        }

        Ok((all_positions, all_boxes))
    }

    /// Current frame index.
    pub fn current_frame(&self) -> usize {
        self.current_frame
    }
}

// ============================================================================
// Internal Implementation
// ============================================================================

/// Read a 4-byte integer (little-endian).
fn read_i32<R: Read>(reader: &mut R) -> Result<i32, String> {
    let mut buf = [0u8; 4];
    reader
        .read_exact(&mut buf)
        .map_err(|e| format!("Failed to read i32: {}", e))?;
    Ok(i32::from_le_bytes(buf))
}

/// Read a 4-byte float (little-endian).
fn read_f32<R: Read>(reader: &mut R) -> Result<f32, String> {
    let mut buf = [0u8; 4];
    reader
        .read_exact(&mut buf)
        .map_err(|e| format!("Failed to read f32: {}", e))?;
    Ok(f32::from_le_bytes(buf))
}

/// Read an 8-byte float (little-endian).
fn read_f64<R: Read>(reader: &mut R) -> Result<f64, String> {
    let mut buf = [0u8; 8];
    reader
        .read_exact(&mut buf)
        .map_err(|e| format!("Failed to read f64: {}", e))?;
    Ok(f64::from_le_bytes(buf))
}

/// Read a Fortran record block marker (4 bytes).
fn read_block_size<R: Read>(reader: &mut R, is_big_endian: bool) -> Result<i32, String> {
    if is_big_endian {
        read_i32_be(reader)
    } else {
        read_i32(reader)
    }
}

/// Read a 4-byte integer (big-endian).
fn read_i32_be<R: Read>(reader: &mut R) -> Result<i32, String> {
    let mut buf = [0u8; 4];
    reader
        .read_exact(&mut buf)
        .map_err(|e| format!("Failed to read i32: {}", e))?;
    Ok(i32::from_be_bytes(buf))
}

/// Detect endianness by checking the first block size (should be 84).
fn detect_endianness<R: Read + Seek>(reader: &mut R) -> Result<bool, String> {
    let mut buf = [0u8; 4];
    reader
        .read_exact(&mut buf)
        .map_err(|e| format!("Failed to read block size: {}", e))?;

    let le_val = i32::from_le_bytes(buf);
    let be_val = i32::from_be_bytes(buf);

    // Seek back to start
    reader
        .seek(SeekFrom::Current(-4))
        .map_err(|e| format!("Failed to seek: {}", e))?;

    // The first block size should be 84
    if le_val == 84 {
        Ok(false) // Little-endian
    } else if be_val == 84 {
        Ok(true) // Big-endian
    } else {
        Err(format!(
            "Invalid DCD header: first block size is {} (LE) or {} (BE), expected 84",
            le_val, be_val
        ))
    }
}

/// Read the DCD header.
fn read_dcd_header_internal<R: Read + Seek>(reader: &mut R) -> Result<DcdHeader, String> {
    // Detect endianness
    let is_big_endian = detect_endianness(reader)?;

    // Helper macro to read i32 with correct endianness
    macro_rules! read_int {
        ($reader:expr) => {
            if is_big_endian {
                read_i32_be($reader)?
            } else {
                read_i32($reader)?
            }
        };
    }

    // First block: main header
    let block1_size = read_int!(reader);
    if block1_size != 84 {
        return Err(format!("Invalid DCD header size: {}", block1_size));
    }

    // Magic number "CORD"
    let mut magic = [0u8; 4];
    reader
        .read_exact(&mut magic)
        .map_err(|e| format!("Failed to read magic: {}", e))?;
    if &magic != b"CORD" {
        return Err(format!("Invalid DCD magic: {:?}", magic));
    }

    // Read header values
    let n_frames_i32 = read_int!(reader);
    if n_frames_i32 < 0 {
        return Err(format!("Invalid number of frames: {}", n_frames_i32));
    }
    let n_frames = n_frames_i32 as usize;

    let start_timestep = read_int!(reader);
    let timestep_interval = read_int!(reader);
    let _n_timesteps = read_int!(reader);
    let _unused1 = read_int!(reader);
    let _unused2 = read_int!(reader);
    let _unused3 = read_int!(reader);
    let _unused4 = read_int!(reader);
    let _n_fixed = read_int!(reader);

    // Timestep is always stored as little-endian float in most implementations
    let timestep = read_f32(reader)?;
    let has_unit_cell = read_int!(reader) != 0;

    // Read CHARMM version info
    let mut is_charmm = false;
    let mut is_64bit = false;

    // Skip to CHARMM version position
    let mut skip_buf = [0u8; 32];
    reader
        .read_exact(&mut skip_buf)
        .map_err(|e| format!("Failed to skip: {}", e))?;

    let charmm_version = read_int!(reader);
    if charmm_version != 0 {
        is_charmm = true;
        // Check for 64-bit flag in position 20 (already read)
        is_64bit = skip_buf[16..20] != [0, 0, 0, 0];
    }

    // End of first block
    let _block1_end = read_int!(reader);

    // Second block: titles
    let block2_size = read_int!(reader);
    if block2_size < 4 {
        return Err(format!("Invalid title block size: {}", block2_size));
    }

    let n_titles = read_int!(reader);
    if n_titles < 0 {
        return Err(format!("Invalid number of titles: {}", n_titles));
    }

    // Validate that block size is consistent with number of titles
    let expected_title_bytes = 4 + n_titles * 80;
    if expected_title_bytes > block2_size {
        return Err(format!(
            "Title block size mismatch: block says {} bytes but {} titles need {} bytes",
            block2_size, n_titles, expected_title_bytes
        ));
    }

    let mut titles = Vec::with_capacity(n_titles as usize);
    for _ in 0..n_titles {
        let mut title_buf = [0u8; 80];
        reader
            .read_exact(&mut title_buf)
            .map_err(|e| format!("Failed to read title: {}", e))?;
        let title = String::from_utf8_lossy(&title_buf).trim().to_string();
        titles.push(title);
    }

    // Skip any remaining bytes in the title block
    let title_bytes_read = 4 + n_titles * 80;
    let remaining = block2_size - title_bytes_read;
    if remaining > 0 {
        let mut skip = vec![0u8; remaining as usize];
        reader.read_exact(&mut skip).ok();
    }
    let _block2_end = read_int!(reader);

    // Third block: number of atoms
    let _block3_size = read_int!(reader);
    let n_atoms_i32 = read_int!(reader);
    if n_atoms_i32 < 0 {
        return Err(format!("Invalid number of atoms: {}", n_atoms_i32));
    }
    let n_atoms = n_atoms_i32 as usize;
    if n_atoms > 100_000_000 {
        return Err(format!(
            "Atom count {} exceeds maximum supported (100,000,000)",
            n_atoms
        ));
    }
    let _block3_end = read_int!(reader);

    // Record position for frame data
    let first_frame_offset = reader
        .stream_position()
        .map_err(|e| format!("Failed to get position: {}", e))?;

    // Calculate frame size with overflow checks
    let coord_size = n_atoms.checked_mul(4)
        .ok_or_else(|| format!("Frame size overflow: n_atoms={}", n_atoms))?;
    let coord_block_size = coord_size.checked_add(8)
        .ok_or_else(|| format!("Frame size overflow: coord_size={}", coord_size))?;
    let unit_cell_size = if has_unit_cell { 4 + 48 + 4 } else { 0 }; // 6 doubles
    let frame_size = coord_block_size.checked_mul(3)
        .and_then(|v| v.checked_add(unit_cell_size))
        .ok_or_else(|| format!("Frame size overflow: coord_block_size={}", coord_block_size))?;

    Ok(DcdHeader {
        n_frames,
        n_atoms,
        start_timestep,
        timestep_interval,
        timestep,
        has_unit_cell,
        is_charmm,
        is_64bit,
        is_big_endian,
        titles,
        first_frame_offset,
        frame_size,
    })
}

/// Read a single frame from the current position.
fn read_frame_internal<R: Read>(
    reader: &mut R,
    header: &DcdHeader,
) -> Result<(Vec<[f64; 3]>, Option<[f64; 6]>), String> {
    let n_atoms = header.n_atoms;
    let is_big_endian = header.is_big_endian;

    // Read unit cell if present
    let box_info = if header.has_unit_cell {
        let _block_size = read_block_size(reader, is_big_endian)?;
        let a = read_f64(reader)?;
        let gamma = read_f64(reader)?;
        let b = read_f64(reader)?;
        let beta = read_f64(reader)?;
        let alpha = read_f64(reader)?;
        let c = read_f64(reader)?;
        let _block_end = read_block_size(reader, is_big_endian)?;

        // Convert box lengths to nm, keep angles in degrees
        Some([
            a * ANGSTROM_TO_NM,
            b * ANGSTROM_TO_NM,
            c * ANGSTROM_TO_NM,
            alpha,
            beta,
            gamma,
        ])
    } else {
        None
    };

    // Read X coordinates
    let _x_block = read_block_size(reader, is_big_endian)?;
    let x_coords: Vec<f32> = (0..n_atoms)
        .map(|_| read_f32(reader))
        .collect::<Result<Vec<_>, _>>()?;
    let _x_end = read_block_size(reader, is_big_endian)?;

    // Read Y coordinates
    let _y_block = read_block_size(reader, is_big_endian)?;
    let y_coords: Vec<f32> = (0..n_atoms)
        .map(|_| read_f32(reader))
        .collect::<Result<Vec<_>, _>>()?;
    let _y_end = read_block_size(reader, is_big_endian)?;

    // Read Z coordinates
    let _z_block = read_block_size(reader, is_big_endian)?;
    let z_coords: Vec<f32> = (0..n_atoms)
        .map(|_| read_f32(reader))
        .collect::<Result<Vec<_>, _>>()?;
    let _z_end = read_block_size(reader, is_big_endian)?;

    // Combine into positions array (converting to nm)
    let positions: Vec<[f64; 3]> = (0..n_atoms)
        .map(|i| {
            [
                x_coords[i] as f64 * ANGSTROM_TO_NM,
                y_coords[i] as f64 * ANGSTROM_TO_NM,
                z_coords[i] as f64 * ANGSTROM_TO_NM,
            ]
        })
        .collect();

    Ok((positions, box_info))
}

// ============================================================================
// Public API
// ============================================================================

/// Read the header from a DCD file.
pub fn read_dcd_header<P: AsRef<Path>>(path: P) -> Result<DcdHeader, String> {
    let file = File::open(path.as_ref()).map_err(|e| format!("Failed to open DCD file: {}", e))?;
    let mut reader = BufReader::new(file);
    read_dcd_header_internal(&mut reader)
}

/// Read a specific frame from a DCD file.
#[allow(dead_code)]
pub fn read_dcd_frame<P: AsRef<Path>>(
    path: P,
    frame: usize,
) -> Result<(Vec<[f64; 3]>, Option<[f64; 6]>), String> {
    let mut reader = DcdReader::open(path)?;
    reader.seek_frame(frame)?;
    reader
        .read_frame()?
        .ok_or_else(|| "Frame not found".to_string())
}
