"""DCD trajectory reader example - read binary trajectory files."""

import numpy as np
from rust_simulation_tools import DcdReader, read_dcd_header

# Quick header inspection
header = read_dcd_header("trajectory.dcd")
print(f"Trajectory: {header['n_frames']} frames, {header['n_atoms']} atoms")
print(f"Has unit cell: {header['has_unit_cell']}")

# Open trajectory for reading
reader = DcdReader("trajectory.dcd")

# Read single frame
positions, box = reader.read_frame()
print(f"\nFrame 0: positions shape {positions.shape}, dtype {positions.dtype}")
if box is not None:
    print(f"Box: {box[:3]} nm, angles {box[3:]} deg")

# Read specific frame
positions_10, _ = reader.read_frame_at(10)
print(f"\nFrame 10: {positions_10[:3]}")  # First 3 atom positions

# Iterate through trajectory
print(f"\nReading all {reader.n_frames} frames...")
reader.seek(0)  # Reset to beginning
frame_count = 0
while True:
    frame = reader.read_frame()
    if frame is None:
        break
    frame_count += 1
print(f"Read {frame_count} frames")

# Read entire trajectory at once (for smaller files)
reader.seek(0)
all_positions, all_boxes = reader.read_all()
trajectory = all_positions.reshape(reader.n_frames, reader.n_atoms, 3)
print(f"\nFull trajectory shape: {trajectory.shape}")
