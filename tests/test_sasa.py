"""Tests for SASA (Solvent Accessible Surface Area) calculations."""

import numpy as np
import pytest
from rust_simulation_tools import (
    calculate_sasa,
    calculate_residue_sasa,
    calculate_total_sasa,
    calculate_sasa_trajectory,
    get_vdw_radius,
    get_radii_array,
)


class TestImports:
    """Test that all SASA functions can be imported."""

    def test_imports(self):
        """Test that all functions are importable."""
        assert callable(calculate_sasa)
        assert callable(calculate_residue_sasa)
        assert callable(calculate_total_sasa)
        assert callable(calculate_sasa_trajectory)
        assert callable(get_vdw_radius)
        assert callable(get_radii_array)


class TestVdwRadii:
    """Test VDW radius lookup functions."""

    def test_get_vdw_radius_common_elements(self):
        """Test VDW radius lookup for common elements."""
        test_cases = [
            ('C', 1.70),
            ('N', 1.55),
            ('O', 1.52),
            ('H', 1.20),
            ('S', 1.80),
            ('P', 1.80),
        ]
        for element, expected in test_cases:
            radius = get_vdw_radius(element)
            assert radius == expected, f"{element} radius {radius} != expected {expected}"

    def test_get_vdw_radius_case_insensitive(self):
        """Test that element lookup is case insensitive."""
        assert get_vdw_radius('c') == get_vdw_radius('C')
        assert get_vdw_radius('n') == get_vdw_radius('N')
        assert get_vdw_radius('o') == get_vdw_radius('O')

    def test_get_radii_array(self):
        """Test batch radius lookup with get_radii_array."""
        elements = ['C', 'N', 'O', 'H', 'S']
        radii = get_radii_array(elements)

        assert len(radii) == len(elements)
        assert radii[0] == 1.70  # C
        assert radii[1] == 1.55  # N
        assert radii[2] == 1.52  # O
        assert radii[3] == 1.20  # H
        assert radii[4] == 1.80  # S

    def test_get_radii_array_mixed_case(self):
        """Test that get_radii_array handles mixed case."""
        elements = ['C', 'c', 'N', 'n']
        radii = get_radii_array(elements)

        assert radii[0] == radii[1]  # C == c
        assert radii[2] == radii[3]  # N == n


class TestSingleAtomSasa:
    """Test SASA calculations for isolated atoms."""

    def test_single_atom_sasa(self):
        """Test SASA of a single isolated atom matches analytical result."""
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([get_vdw_radius('C')])  # 1.7 Angstrom
        residue_indices = np.array([0])

        result = calculate_sasa(coords, radii, residue_indices)

        # Analytical: 4*pi*(vdw_radius + probe_radius)^2 = 4*pi*(1.7 + 1.4)^2 ≈ 120.76 A^2
        expected = 4 * np.pi * (1.7 + 1.4)**2
        error = abs(result['total'] - expected) / expected * 100

        assert error < 5.0, f"Error {error:.2f}% exceeds 5% threshold"

    def test_two_distant_atoms(self):
        """Test SASA of two atoms far apart (no interaction)."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0]
        ])
        radii = np.array([1.7, 1.7])
        residue_indices = np.array([0, 1])

        result = calculate_sasa(coords, radii, residue_indices)

        # Each atom should have full SASA
        single_atom_sasa = 4 * np.pi * (1.7 + 1.4)**2
        expected = 2 * single_atom_sasa
        error = abs(result['total'] - expected) / expected * 100

        assert error < 5.0, f"Error {error:.2f}% exceeds 5% threshold"


class TestSasaOcclusion:
    """Test SASA calculations with atomic occlusion."""

    def test_two_close_atoms_have_reduced_sasa(self):
        """Test that two close atoms have less SASA than two distant atoms."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0]  # Close enough for partial occlusion
        ])
        radii = np.array([1.7, 1.7])
        residue_indices = np.array([0, 1])

        result = calculate_sasa(coords, radii, residue_indices)

        single_atom_sasa = 4 * np.pi * (1.7 + 1.4)**2
        max_total = 2 * single_atom_sasa

        assert result['total'] < max_total * 0.95, \
            f"No occlusion detected: {result['total']:.2f} >= {max_total * 0.95:.2f}"

    def test_buried_atom_has_less_sasa(self):
        """Test that a central atom surrounded by neighbors is more buried."""
        # Central atom surrounded by 6 neighbors
        coords = np.array([
            [0.0, 0.0, 0.0],    # Central atom
            [3.5, 0.0, 0.0],
            [-3.5, 0.0, 0.0],
            [0.0, 3.5, 0.0],
            [0.0, -3.5, 0.0],
            [0.0, 0.0, 3.5],
            [0.0, 0.0, -3.5],
        ])
        radii = np.full(7, 1.7)
        residue_indices = np.arange(7)

        result = calculate_sasa(coords, radii, residue_indices)

        central_sasa = result['per_atom'][0]
        peripheral_mean = result['per_atom'][1:].mean()

        assert central_sasa < peripheral_mean, \
            f"Central atom SASA {central_sasa:.2f} should be < peripheral mean {peripheral_mean:.2f}"


class TestPerResidueSasa:
    """Test per-residue SASA aggregation."""

    def test_per_residue_equals_sum_of_atoms(self):
        """Test that per-residue SASA equals sum of per-atom SASA."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ])
        radii = np.array([1.7, 1.5, 1.7])
        residue_indices = np.array([0, 0, 1])  # First two in res 0, third in res 1

        result = calculate_sasa(coords, radii, residue_indices)

        res0_sasa = result['per_residue'][0]
        res1_sasa = result['per_residue'][1]
        atom0_sasa = result['per_atom'][0]
        atom1_sasa = result['per_atom'][1]
        atom2_sasa = result['per_atom'][2]

        assert abs(res0_sasa - (atom0_sasa + atom1_sasa)) < 0.01, \
            f"Residue 0: {res0_sasa} != {atom0_sasa} + {atom1_sasa}"
        assert abs(res1_sasa - atom2_sasa) < 0.01, \
            f"Residue 1: {res1_sasa} != {atom2_sasa}"


class TestSasaFunctions:
    """Test different SASA calculation functions give consistent results."""

    def test_functions_consistency(self):
        """Test that calculate_sasa, calculate_residue_sasa, and calculate_total_sasa are consistent."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ])
        radii = np.array([1.7, 1.7])
        residue_indices = np.array([0, 1])

        full_result = calculate_sasa(coords, radii, residue_indices)
        residue_result = calculate_residue_sasa(coords, radii, residue_indices)
        total_result = calculate_total_sasa(coords, radii)

        assert abs(full_result['total'] - total_result) < 0.01, \
            f"Total mismatch: {full_result['total']} != {total_result}"
        assert abs(full_result['per_residue'][0] - residue_result[0]) < 0.01, \
            f"Residue 0 mismatch"
        assert abs(full_result['per_residue'][1] - residue_result[1]) < 0.01, \
            f"Residue 1 mismatch"


class TestProbeRadius:
    """Test different probe radii."""

    def test_sasa_increases_with_probe_radius(self):
        """Test that SASA increases with larger probe radius."""
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([1.5])

        probe_radii = [0.0, 1.0, 1.4, 2.0]
        results = []

        for probe_r in probe_radii:
            sasa = calculate_total_sasa(coords, radii, probe_radius=probe_r)
            results.append(sasa)

        for i in range(len(results) - 1):
            assert results[i] < results[i+1], \
                f"SASA should increase with probe radius: {results[i]} >= {results[i+1]}"


class TestSpherePoints:
    """Test different sphere point densities."""

    def test_sphere_point_convergence(self):
        """Test that results converge with more sphere points."""
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([1.5])

        point_counts = [92, 162, 242, 480, 960]
        results = []

        for n_points in point_counts:
            sasa = calculate_total_sasa(coords, radii, n_sphere_points=n_points)
            results.append(sasa)

        # Results should converge - standard deviation should be small
        std = np.std(results)
        assert std < 2.0, f"Poor convergence with std = {std:.2f}"


class TestTrajectory:
    """Test trajectory-based SASA calculations."""

    def test_calculate_sasa_trajectory(self):
        """Test SASA calculation over multiple frames."""
        n_frames = 3
        n_atoms = 5

        # Create random coordinates for each frame
        np.random.seed(42)
        coords = np.random.randn(n_frames, n_atoms, 3) * 5.0
        radii = np.full(n_atoms, 1.7)
        residue_indices = np.arange(n_atoms)

        result = calculate_sasa_trajectory(coords, radii, residue_indices)

        # Result should be a dict with 'per_atom', 'per_residue', 'total'
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'per_atom' in result, "Result should contain 'per_atom'"
        assert 'per_residue' in result, "Result should contain 'per_residue'"
        assert 'total' in result, "Result should contain 'total'"

        # per_atom is returned as a flat array (n_frames * n_atoms,)
        # that can be reshaped to (n_frames, n_atoms)
        per_atom = result['per_atom']
        assert per_atom.shape == (n_frames * n_atoms,), \
            f"Expected per_atom shape ({n_frames * n_atoms},), got {per_atom.shape}"

        # Verify it can be reshaped to (n_frames, n_atoms)
        per_atom_reshaped = per_atom.reshape(n_frames, n_atoms)
        assert per_atom_reshaped.shape == (n_frames, n_atoms)

        # total should have shape (n_frames,)
        total = result['total']
        assert total.shape == (n_frames,), \
            f"Expected total shape ({n_frames},), got {total.shape}"

        # per_residue should be a list of n_frames dicts
        per_residue = result['per_residue']
        assert isinstance(per_residue, list), "per_residue should be a list"
        assert len(per_residue) == n_frames, \
            f"Expected {n_frames} per_residue entries, got {len(per_residue)}"

        # All values should be non-negative
        assert np.all(per_atom >= 0), "SASA values should be non-negative"
        assert np.all(total >= 0), "Total SASA values should be non-negative"

    def test_trajectory_matches_frame_by_frame(self):
        """Test that trajectory SASA matches frame-by-frame calculation."""
        n_frames = 3
        n_atoms = 4

        np.random.seed(123)
        coords = np.random.randn(n_frames, n_atoms, 3) * 5.0
        radii = np.full(n_atoms, 1.6)
        residue_indices = np.arange(n_atoms)

        # Calculate using trajectory function
        traj_result = calculate_sasa_trajectory(coords, radii, residue_indices)

        # Reshape per_atom to (n_frames, n_atoms)
        traj_per_atom = traj_result['per_atom'].reshape(n_frames, n_atoms)

        # Calculate frame by frame
        for frame_idx in range(n_frames):
            frame_coords = coords[frame_idx]
            single_result = calculate_sasa(frame_coords, radii, residue_indices)

            # Per-atom SASA should match
            np.testing.assert_allclose(
                traj_per_atom[frame_idx],
                single_result['per_atom'],
                rtol=1e-5,
                err_msg=f"Frame {frame_idx} per-atom SASA mismatch"
            )

            # Total SASA should match
            np.testing.assert_allclose(
                traj_result['total'][frame_idx],
                single_result['total'],
                rtol=1e-5,
                err_msg=f"Frame {frame_idx} total SASA mismatch"
            )


class TestPerformance:
    """Test performance on larger systems."""

    @pytest.mark.parametrize("n_atoms", [100, 500])
    def test_performance_scales(self, n_atoms):
        """Test that SASA calculation handles larger systems."""
        np.random.seed(42)
        coords = np.random.randn(n_atoms, 3) * 10.0
        radii = np.random.uniform(1.2, 1.9, n_atoms)
        residue_indices = np.arange(n_atoms)

        result = calculate_sasa(coords, radii, residue_indices, n_sphere_points=480)

        assert result['total'] > 0, "Total SASA should be positive"
        assert len(result['per_atom']) == n_atoms
        assert len(result['per_residue']) == n_atoms


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
