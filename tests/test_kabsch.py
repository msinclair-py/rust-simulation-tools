import numpy as np
import pytest
from rust_simulation_tools import kabsch_align


class TestKabschAlign:
    """Test suite for Kabsch alignment implementation."""
    
    @pytest.fixture
    def simple_triangle(self):
        """Create a simple 3-atom triangle reference structure."""
        return np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=np.float64)
    
    def test_rotation_only(self, simple_triangle):
        """Test alignment with pure rotation (90 degrees around z-axis)."""
        reference = simple_triangle
        
        # Rotate by 90 degrees around z-axis
        theta = np.pi / 2
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        
        mobile = reference @ rotation_matrix.T
        trajectory = mobile.reshape(1, 3, 3)
        align_indices = np.array([0, 1, 2], dtype=np.uintp)
        
        aligned = kabsch_align(trajectory, reference, align_indices)
        
        rmsd = np.sqrt(np.mean(np.sum((aligned[0] - reference)**2, axis=1)))
        
        assert rmsd < 1e-6, f"RMSD after alignment should be near zero, got {rmsd}"
    
    def test_translation_only(self, simple_triangle):
        """Test alignment with pure translation."""
        reference = simple_triangle
        translation = np.array([5.0, 3.0, 2.0])
        
        mobile = reference + translation
        trajectory = mobile.reshape(1, 3, 3)
        align_indices = np.array([0, 1, 2], dtype=np.uintp)
        
        aligned = kabsch_align(trajectory, reference, align_indices)
        
        rmsd = np.sqrt(np.mean(np.sum((aligned[0] - reference)**2, axis=1)))
        
        assert rmsd < 1e-6, f"RMSD after alignment should be near zero, got {rmsd}"
    
    def test_rotation_and_translation(self, simple_triangle):
        """Test alignment with both rotation and translation."""
        reference = simple_triangle
        
        # Rotate by 45 degrees around z-axis and translate
        theta = np.pi / 4
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        translation = np.array([10.0, -5.0, 3.0])
        
        mobile = reference @ rotation_matrix.T + translation
        trajectory = mobile.reshape(1, 3, 3)
        align_indices = np.array([0, 1, 2], dtype=np.uintp)
        
        aligned = kabsch_align(trajectory, reference, align_indices)
        
        rmsd = np.sqrt(np.mean(np.sum((aligned[0] - reference)**2, axis=1)))
        
        assert rmsd < 1e-6, f"RMSD after alignment should be near zero, got {rmsd}"
    
    def test_multiple_frames(self):
        """Test alignment with multiple trajectory frames."""
        reference = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0]
        ], dtype=np.float64)
        
        num_frames = 5
        trajectory = np.zeros((num_frames, 4, 3), dtype=np.float64)
        
        # Create different rotations for each frame
        for i in range(num_frames):
            theta = np.pi * i / 4  # 0, 45, 90, 135, 180 degrees
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            translation = np.array([i * 2.0, i * 3.0, i * 1.0])
            trajectory[i] = reference @ rotation_matrix.T + translation
        
        align_indices = np.array([0, 1, 2, 3], dtype=np.uintp)
        
        aligned = kabsch_align(trajectory, reference, align_indices)
        
        # Check each frame
        for i in range(num_frames):
            rmsd = np.sqrt(np.mean(np.sum((aligned[i] - reference)**2, axis=1)))
            assert rmsd < 1e-6, f"Frame {i}: RMSD should be near zero, got {rmsd}"
    
    def test_partial_alignment(self):
        """Test alignment using subset of atoms."""
        # Create a structure with 6 atoms (2 rigid bodies)
        reference = np.array([
            # Rigid body 1 (will be used for alignment)
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            # Rigid body 2 (flexible, not used for alignment)
            [5.0, 5.0, 0.0],
            [6.0, 5.0, 0.0],
            [5.0, 6.0, 0.0]
        ], dtype=np.float64)
        
        # Rotate and translate everything
        theta = np.pi / 3
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        translation = np.array([3.0, 4.0, 1.0])
        
        mobile = reference @ rotation_matrix.T + translation
        trajectory = mobile.reshape(1, 6, 3)
        
        # Only align using first 3 atoms
        align_indices = np.array([0, 1, 2], dtype=np.uintp)
        
        aligned = kabsch_align(trajectory, reference, align_indices)
        
        # First 3 atoms should align perfectly
        rmsd_aligned_atoms = np.sqrt(np.mean(np.sum(
            (aligned[0][:3] - reference[:3])**2, axis=1
        )))
        assert rmsd_aligned_atoms < 1e-6, \
            f"Aligned atoms RMSD should be near zero, got {rmsd_aligned_atoms}"
        
        # Last 3 atoms should also be transformed (but may not match reference perfectly)
        # They should at least be different from the mobile coordinates
        assert not np.allclose(aligned[0][3:], mobile[3:]), \
            "Non-aligned atoms should still be transformed"
    
    def test_identity_alignment(self, simple_triangle):
        """Test that aligning identical structures gives identity transformation."""
        reference = simple_triangle
        mobile = reference.copy()
        
        trajectory = mobile.reshape(1, 3, 3)
        align_indices = np.array([0, 1, 2], dtype=np.uintp)
        
        aligned = kabsch_align(trajectory, reference, align_indices)
        
        # Should be identical (within numerical precision)
        assert np.allclose(aligned[0], reference, atol=1e-10), \
            "Identity alignment should return identical coordinates"
    
    def test_rmsd_improvement(self):
        """Test that alignment always improves (or maintains) RMSD."""
        reference = np.random.rand(10, 3).astype(np.float64)
        
        # Random rotation and translation
        theta = np.random.rand() * 2 * np.pi
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        translation = np.random.rand(3) * 10
        
        mobile = reference @ rotation_matrix.T + translation
        trajectory = mobile.reshape(1, 10, 3)
        align_indices = np.arange(10, dtype=np.uintp)
        
        aligned = kabsch_align(trajectory, reference, align_indices)
        
        rmsd_before = np.sqrt(np.mean(np.sum((mobile - reference)**2, axis=1)))
        rmsd_after = np.sqrt(np.mean(np.sum((aligned[0] - reference)**2, axis=1)))
        
        assert rmsd_after <= rmsd_before, \
            f"RMSD should improve: before={rmsd_before:.6f}, after={rmsd_after:.6f}"
        assert rmsd_after < 1e-6, \
            f"RMSD should be near zero for pure rotation+translation, got {rmsd_after}"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
