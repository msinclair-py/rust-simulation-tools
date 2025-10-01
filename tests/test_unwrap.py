import numpy as np
import pytest
from rust_simulation_tools import unwrap_system

class TestUnwrapSystem:
    """Test suite for trajectory unwrapping implementation."""

    def test_no_unwrapping_needed(self):
        """Test that coordinates inside box remain unchanged."""
        # Single molecule that doesn't cross boundaries
        trajectory = np.array([[
            [1.0, 1.0, 1.0],
            [1.5, 1.0, 1.0],
            [1.0, 1.5, 1.0],
        ]], dtype=np.float64)

        box = np.array([[10.0, 10.0, 10.0]], dtype=np.float64)
        fragments = np.array([0, 0, 0], dtype=np.uintp)

        unwrapped = unwrap_system(trajectory, box, fragments)

        assert np.allclose(unwrapped[0], trajectory[0]), \
            "Coordinates not crossing boundaries should remain unchanged"

    def test_simple_boundary_crossing(self):
        """Test unwrapping when molecule crosses periodic boundary."""
        # Create a molecule that crosses the boundary between frames
        box_size = 10.0

        # Frame 0: molecule near right boundary
        frame0 = np.array([
            [9.0, 5.0, 5.0],
            [9.5, 5.0, 5.0],
        ], dtype=np.float64)

        # Frame 1: molecule wrapped to left side (crossed boundary)
        frame1 = np.array([
            [0.2, 5.0, 5.0],  # Wrapped from 10.2
            [0.7, 5.0, 5.0],  # Wrapped from 10.7
        ], dtype=np.float64)

        trajectory = np.array([frame0, frame1], dtype=np.float64)
        box = np.array([[box_size, box_size, box_size]] * 2, dtype=np.float64)
        fragments = np.array([0, 0], dtype=np.uintp)

        unwrapped = unwrap_system(trajectory, box, fragments)

        # Frame 0 should be unchanged
        assert np.allclose(unwrapped[0], frame0)

        # Frame 1 should be unwrapped (shifted by +box_size in x)
        expected_frame1 = frame1 + np.array([box_size, 0, 0])
        assert np.allclose(unwrapped[1], expected_frame1, atol=1e-10), \
            f"Expected {expected_frame1}, got {unwrapped[1]}"

    def test_multiple_frames_crossing(self):
        """Test unwrapping across multiple frames."""
        box_size = 10.0

        # Molecule moving continuously, crossing boundary multiple times
        trajectory = np.array([
            [[2.0, 5.0, 5.0]],  # Frame 0
            [[6.0, 5.0, 5.0]],  # Frame 1: moved right (delta=4.0)
            [[0.0, 5.0, 5.0]],  # Frame 2: wrapped (delta=-6.0 < -5.0, actually at 10.0)
            [[4.0, 5.0, 5.0]],  # Frame 3: moved right (delta=4.0, actually at 14.0)
            [[8.0, 5.0, 5.0]],  # Frame 4: moved right (delta=4.0, actually at 18.0)
        ], dtype=np.float64)

        box = np.array([[box_size, box_size, box_size]] * 5, dtype=np.float64)
        fragments = np.array([0], dtype=np.uintp)

        unwrapped = unwrap_system(trajectory, box, fragments)

        # Check that x-coordinates are monotonically increasing
        x_coords = unwrapped[:, 0, 0]
        assert np.all(np.diff(x_coords) > 0), \
            f"Unwrapped coordinates should increase monotonically, got {x_coords}"

    def test_multiple_fragments(self):
        """Test that different fragments can unwrap independently."""
        box_size = 10.0

        # Two separate molecules
        trajectory = np.array([
            # Frame 0
            [[2.0, 5.0, 5.0],  # Molecule 0 atom 0
             [6.0, 5.0, 5.0],  # Molecule 0 atom 1
             [7.0, 5.0, 5.0],  # Molecule 1 atom 0
             [8.0, 5.0, 5.0]], # Molecule 1 atom 1
            # Frame 1: Molecule 0 atom 1 crosses boundary, others don't
            [[2.5, 5.0, 5.0],  # Moved slightly right (delta=0.5)
             [0.0, 5.0, 5.0],  # Wrapped (delta=-6.0 < -5.0, actually at 10.0)
             [7.5, 5.0, 5.0],  # Didn't cross (delta=0.5)
             [8.5, 5.0, 5.0]], # Didn't cross (delta=0.5)
        ], dtype=np.float64)

        box = np.array([[box_size, box_size, box_size]] * 2, dtype=np.float64)
        fragments = np.array([0, 0, 1, 1], dtype=np.uintp)

        unwrapped = unwrap_system(trajectory, box, fragments)

        # Check molecule 0 atom 0 moved normally
        assert 2.0 < unwrapped[1, 0, 0] < 3.0, \
            f"Molecule 0 atom 0 should not need unwrapping, got {unwrapped[1, 0, 0]}"

        # Molecule 0 atom 1 should be unwrapped in frame 1
        assert unwrapped[1, 1, 0] >= box_size, \
            f"Molecule 0 atom 1 should be unwrapped (at ~10.0), got {unwrapped[1, 1, 0]}"

        # Molecule 1 should remain mostly unchanged
        assert unwrapped[1, 2, 0] < box_size, \
            "Molecule 1 should not be unwrapped"
        assert unwrapped[1, 3, 0] < box_size, \
            "Molecule 1 should not be unwrapped"

    def test_3d_unwrapping(self):
        """Test unwrapping in all three dimensions."""
        box_size = 10.0

        trajectory = np.array([
            [[5.0, 5.0, 5.0]],
            [[9.5, 9.5, 9.5]],  # Move near boundaries
            [[0.5, 0.5, 0.5]],  # Crossed boundaries in all dimensions (actually at ~10.5)
        ], dtype=np.float64)

        box = np.array([[box_size, box_size, box_size]] * 3, dtype=np.float64)
        fragments = np.array([0], dtype=np.uintp)

        unwrapped = unwrap_system(trajectory, box, fragments)

        # All dimensions of frame 2 should be unwrapped
        assert np.all(unwrapped[2, 0] > box_size), \
            f"All dimensions should be unwrapped, got {unwrapped[2, 0]}"

    def test_input_validation(self):
        """Test that invalid inputs raise appropriate errors."""
        trajectory = np.array([[[1.0, 1.0, 1.0]]], dtype=np.float64)
        box = np.array([[10.0, 10.0, 10.0]], dtype=np.float64)
        fragments = np.array([0], dtype=np.uintp)

        # Wrong number of frames in box
        with pytest.raises(Exception):
            bad_box = np.array([[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]], dtype=np.float64)
            unwrap_system(trajectory, bad_box, fragments)

        # Wrong number of fragment indices
        with pytest.raises(Exception):
            bad_fragments = np.array([0, 1], dtype=np.uintp)
            unwrap_system(trajectory, box, bad_fragments)

        # Invalid box dimensions shape
        with pytest.raises(Exception):
            bad_box = np.array([[10.0, 10.0]], dtype=np.float64)
            unwrap_system(trajectory, bad_box, fragments)

if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', 'tb=short'])
