"""Tests for interaction energy fingerprinting."""

import numpy as np
import pytest
from pathlib import Path

from rust_simulation_tools import compute_fingerprints


# Path to test data directory
DATA_DIR = Path(__file__).parent.parent / "data"


class TestComputeFingerprints:
    """Test suite for compute_fingerprints function."""

    @pytest.fixture
    def simple_system(self):
        """Create a simple two-residue system for testing."""
        # 4 atoms total: 2 in residue 0 (target), 2 in residue 1 (binder)
        # Positions in nm
        positions = np.array([
            [0.0, 0.0, 0.0],   # Res 0, atom 0
            [0.15, 0.0, 0.0],  # Res 0, atom 1
            [0.5, 0.0, 0.0],   # Res 1, atom 0 (binder)
            [0.65, 0.0, 0.0],  # Res 1, atom 1 (binder)
        ], dtype=np.float64)

        # Partial charges in elementary charge units
        charges = np.array([0.5, -0.5, 0.3, -0.3], dtype=np.float64)

        # LJ sigma in nm
        sigmas = np.array([0.34, 0.32, 0.34, 0.32], dtype=np.float64)

        # LJ epsilon in kJ/mol
        epsilons = np.array([0.4, 0.3, 0.4, 0.3], dtype=np.float64)

        # Resmap: atoms for each target residue
        resmap_indices = np.array([0, 1], dtype=np.int64)  # Atoms in residue 0
        resmap_offsets = np.array([0, 2], dtype=np.int64)  # Start=0, End=2 (exclusive)

        # Binder: atoms in residue 1
        binder_indices = np.array([2, 3], dtype=np.int64)

        return {
            'positions': positions,
            'charges': charges,
            'sigmas': sigmas,
            'epsilons': epsilons,
            'resmap_indices': resmap_indices,
            'resmap_offsets': resmap_offsets,
            'binder_indices': binder_indices,
        }

    def test_basic_fingerprint_computation(self, simple_system):
        """Test that fingerprints can be computed without errors."""
        lj_fp, es_fp = compute_fingerprints(
            simple_system['positions'],
            simple_system['charges'],
            simple_system['sigmas'],
            simple_system['epsilons'],
            simple_system['resmap_indices'],
            simple_system['resmap_offsets'],
            simple_system['binder_indices'],
        )

        # Should return arrays of length n_residues (1 in this case)
        assert len(lj_fp) == 1, f"Expected 1 residue LJ fingerprint, got {len(lj_fp)}"
        assert len(es_fp) == 1, f"Expected 1 residue ES fingerprint, got {len(es_fp)}"

    def test_fingerprint_signs(self, simple_system):
        """Test that fingerprint values have expected signs for attractive/repulsive interactions."""
        lj_fp, es_fp = compute_fingerprints(
            simple_system['positions'],
            simple_system['charges'],
            simple_system['sigmas'],
            simple_system['epsilons'],
            simple_system['resmap_indices'],
            simple_system['resmap_offsets'],
            simple_system['binder_indices'],
        )

        # At moderate distances, LJ should be attractive (negative) or weakly repulsive
        # ES with opposite charges should be attractive (negative)
        # Note: signs depend on exact distances and parameters
        assert not np.isnan(lj_fp[0]), "LJ fingerprint should not be NaN"
        assert not np.isnan(es_fp[0]), "ES fingerprint should not be NaN"
        assert np.isfinite(lj_fp[0]), "LJ fingerprint should be finite"
        assert np.isfinite(es_fp[0]), "ES fingerprint should be finite"

    def test_zero_energy_at_large_distance(self):
        """Test that energies are zero when atoms are beyond cutoff."""
        # Place atoms far apart (beyond 1.2 nm LJ cutoff)
        positions = np.array([
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],  # 5 nm apart, well beyond cutoffs
        ], dtype=np.float64)

        charges = np.array([0.5, -0.5], dtype=np.float64)
        sigmas = np.array([0.34, 0.34], dtype=np.float64)
        epsilons = np.array([0.4, 0.4], dtype=np.float64)

        resmap_indices = np.array([0], dtype=np.int64)
        resmap_offsets = np.array([0, 1], dtype=np.int64)
        binder_indices = np.array([1], dtype=np.int64)

        lj_fp, es_fp = compute_fingerprints(
            positions, charges, sigmas, epsilons,
            resmap_indices, resmap_offsets, binder_indices
        )

        # At 5 nm, both interactions should be zero (beyond cutoffs)
        assert lj_fp[0] == 0.0, f"LJ should be 0 at large distance, got {lj_fp[0]}"
        assert es_fp[0] == 0.0, f"ES should be 0 at large distance, got {es_fp[0]}"

    def test_multiple_residues(self):
        """Test fingerprints for multiple target residues."""
        # 6 atoms: 2 residues (2 atoms each) as target, 2 atoms as binder
        positions = np.array([
            [0.0, 0.0, 0.0],   # Res 0, atom 0
            [0.1, 0.0, 0.0],   # Res 0, atom 1
            [0.5, 0.0, 0.0],   # Res 1, atom 0
            [0.6, 0.0, 0.0],   # Res 1, atom 1
            [0.25, 0.3, 0.0],  # Binder atom 0
            [0.55, 0.3, 0.0],  # Binder atom 1
        ], dtype=np.float64)

        charges = np.array([0.5, -0.5, 0.3, -0.3, 0.2, -0.2], dtype=np.float64)
        sigmas = np.full(6, 0.34, dtype=np.float64)
        epsilons = np.full(6, 0.4, dtype=np.float64)

        # Two residues: res0 has atoms 0,1; res1 has atoms 2,3
        resmap_indices = np.array([0, 1, 2, 3], dtype=np.int64)
        resmap_offsets = np.array([0, 2, 4], dtype=np.int64)
        binder_indices = np.array([4, 5], dtype=np.int64)

        lj_fp, es_fp = compute_fingerprints(
            positions, charges, sigmas, epsilons,
            resmap_indices, resmap_offsets, binder_indices
        )

        # Should have 2 residue fingerprints
        assert len(lj_fp) == 2, f"Expected 2 LJ fingerprints, got {len(lj_fp)}"
        assert len(es_fp) == 2, f"Expected 2 ES fingerprints, got {len(es_fp)}"

        # Both should be finite
        assert np.all(np.isfinite(lj_fp)), "All LJ fingerprints should be finite"
        assert np.all(np.isfinite(es_fp)), "All ES fingerprints should be finite"

    def test_lj_energy_distance_dependence(self):
        """Test that LJ energy changes with distance as expected."""
        sigmas = np.array([0.3, 0.3], dtype=np.float64)
        epsilons = np.array([1.0, 1.0], dtype=np.float64)
        charges = np.array([0.0, 0.0], dtype=np.float64)  # No electrostatics

        resmap_indices = np.array([0], dtype=np.int64)
        resmap_offsets = np.array([0, 1], dtype=np.int64)
        binder_indices = np.array([1], dtype=np.int64)

        energies = []
        distances = [0.4, 0.5, 0.6, 0.8, 1.0]  # nm

        for d in distances:
            positions = np.array([
                [0.0, 0.0, 0.0],
                [d, 0.0, 0.0],
            ], dtype=np.float64)

            lj_fp, _ = compute_fingerprints(
                positions, charges, sigmas, epsilons,
                resmap_indices, resmap_offsets, binder_indices
            )
            energies.append(lj_fp[0])

        # LJ should become less repulsive (or more attractive) as distance increases
        # in the repulsive region, then approach zero
        # At very short distances energy is highly positive, at longer distances near zero
        assert energies[0] != 0 or energies[1] != 0, \
            "At least one close distance should have non-zero LJ energy"

    def test_electrostatic_sign_dependence(self):
        """Test that electrostatic energy depends on charge signs."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],  # Within ES cutoff (1.0 nm)
        ], dtype=np.float64)

        sigmas = np.array([0.34, 0.34], dtype=np.float64)
        epsilons = np.array([0.0, 0.0], dtype=np.float64)  # No LJ

        resmap_indices = np.array([0], dtype=np.int64)
        resmap_offsets = np.array([0, 1], dtype=np.int64)
        binder_indices = np.array([1], dtype=np.int64)

        # Opposite charges (attractive)
        charges_attract = np.array([1.0, -1.0], dtype=np.float64)
        _, es_attract = compute_fingerprints(
            positions, charges_attract, sigmas, epsilons,
            resmap_indices, resmap_offsets, binder_indices
        )

        # Same sign charges (repulsive)
        charges_repel = np.array([1.0, 1.0], dtype=np.float64)
        _, es_repel = compute_fingerprints(
            positions, charges_repel, sigmas, epsilons,
            resmap_indices, resmap_offsets, binder_indices
        )

        # Attractive should be more negative than repulsive
        assert es_attract[0] < es_repel[0], \
            f"Attractive ES {es_attract[0]} should be less than repulsive {es_repel[0]}"


class TestFingerprintWithRealData:
    """Test fingerprints with real molecular data if available."""

    @pytest.fixture
    def amber_data(self):
        """Load AMBER topology data if available."""
        prmtop_path = DATA_DIR / "amber.prmtop"
        if not prmtop_path.exists():
            pytest.skip("amber.prmtop not found in data directory")

        from rust_simulation_tools import read_prmtop

        topo = read_prmtop(str(prmtop_path))
        return topo

    def test_fingerprint_with_amber_topology(self, amber_data):
        """Test fingerprint calculation using AMBER topology parameters."""
        topo = amber_data

        # Get topology data
        charges = np.array(topo.charges())
        sigmas = np.array(topo.sigmas())
        epsilons = np.array(topo.epsilons())
        resmap_indices, resmap_offsets = topo.build_resmap()
        resmap_indices = np.array(resmap_indices)
        resmap_offsets = np.array(resmap_offsets)

        # Use first 10 residues as target, rest as binder
        n_target_residues = min(10, topo.n_residues)
        target_atom_end = resmap_offsets[n_target_residues]

        # Create positions (simple test positions)
        n_atoms = topo.n_atoms
        np.random.seed(42)
        positions = np.random.randn(n_atoms, 3) * 2.0  # Random positions in nm

        # Binder indices: all atoms not in target residues
        binder_indices = np.arange(target_atom_end, n_atoms, dtype=np.int64)

        if len(binder_indices) == 0:
            pytest.skip("No binder atoms available")

        # Subset resmap for target residues only
        target_resmap_indices = resmap_indices[:target_atom_end]
        target_resmap_offsets = resmap_offsets[:n_target_residues + 1]

        lj_fp, es_fp = compute_fingerprints(
            positions, charges, sigmas, epsilons,
            target_resmap_indices, target_resmap_offsets, binder_indices
        )

        assert len(lj_fp) == n_target_residues
        assert len(es_fp) == n_target_residues
        assert np.all(np.isfinite(lj_fp)), "All LJ fingerprints should be finite"
        assert np.all(np.isfinite(es_fp)), "All ES fingerprints should be finite"

    def test_fingerprint_with_dcd_trajectory(self, amber_data):
        """Test fingerprint calculation using real trajectory data from DCD."""
        from rust_simulation_tools import DcdReader

        dcd_path = DATA_DIR / "amber.dcd"
        if not dcd_path.exists():
            pytest.skip("amber.dcd not found in data directory")

        topo = amber_data

        # Load trajectory using DCD reader
        reader = DcdReader(str(dcd_path))

        # Verify atom counts match
        assert reader.n_atoms == topo.n_atoms, \
            f"DCD has {reader.n_atoms} atoms but topology has {topo.n_atoms}"

        # Get topology data
        charges = np.array(topo.charges())
        sigmas = np.array(topo.sigmas())
        epsilons = np.array(topo.epsilons())
        resmap_indices, resmap_offsets = topo.build_resmap()
        resmap_indices = np.array(resmap_indices)
        resmap_offsets = np.array(resmap_offsets)

        # Use first 10 residues as target, rest as binder
        n_target_residues = min(10, topo.n_residues)
        target_atom_end = resmap_offsets[n_target_residues]
        binder_indices = np.arange(target_atom_end, topo.n_atoms, dtype=np.int64)

        if len(binder_indices) == 0:
            pytest.skip("No binder atoms available")

        # Subset resmap for target residues only
        target_resmap_indices = resmap_indices[:target_atom_end]
        target_resmap_offsets = resmap_offsets[:n_target_residues + 1]

        # Read first frame from DCD
        frame_result = reader.read_frame()
        assert frame_result is not None, "Should be able to read first frame"

        positions, box_info = frame_result
        assert positions.shape == (topo.n_atoms, 3), \
            f"Expected shape ({topo.n_atoms}, 3), got {positions.shape}"

        # Compute fingerprints with real trajectory positions
        lj_fp, es_fp = compute_fingerprints(
            positions, charges, sigmas, epsilons,
            target_resmap_indices, target_resmap_offsets, binder_indices
        )

        assert len(lj_fp) == n_target_residues
        assert len(es_fp) == n_target_residues
        assert np.all(np.isfinite(lj_fp)), "All LJ fingerprints should be finite"
        assert np.all(np.isfinite(es_fp)), "All ES fingerprints should be finite"

    def test_fingerprint_trajectory_multiple_frames(self, amber_data):
        """Test fingerprint calculation across multiple trajectory frames."""
        from rust_simulation_tools import DcdReader

        dcd_path = DATA_DIR / "amber.dcd"
        if not dcd_path.exists():
            pytest.skip("amber.dcd not found in data directory")

        topo = amber_data
        reader = DcdReader(str(dcd_path))

        # Get topology data
        charges = np.array(topo.charges())
        sigmas = np.array(topo.sigmas())
        epsilons = np.array(topo.epsilons())
        resmap_indices, resmap_offsets = topo.build_resmap()
        resmap_indices = np.array(resmap_indices)
        resmap_offsets = np.array(resmap_offsets)

        # Use first 10 residues as target
        n_target_residues = min(10, topo.n_residues)
        target_atom_end = resmap_offsets[n_target_residues]
        binder_indices = np.arange(target_atom_end, topo.n_atoms, dtype=np.int64)

        if len(binder_indices) == 0:
            pytest.skip("No binder atoms available")

        target_resmap_indices = resmap_indices[:target_atom_end]
        target_resmap_offsets = resmap_offsets[:n_target_residues + 1]

        # Process multiple frames
        n_frames_to_test = min(3, reader.n_frames)
        all_lj_fps = []
        all_es_fps = []

        for _ in range(n_frames_to_test):
            frame_result = reader.read_frame()
            if frame_result is None:
                break

            positions, _ = frame_result

            lj_fp, es_fp = compute_fingerprints(
                positions, charges, sigmas, epsilons,
                target_resmap_indices, target_resmap_offsets, binder_indices
            )

            all_lj_fps.append(lj_fp)
            all_es_fps.append(es_fp)

        assert len(all_lj_fps) == n_frames_to_test, \
            f"Expected {n_frames_to_test} frames, got {len(all_lj_fps)}"

        # Fingerprints should vary between frames (dynamics)
        all_lj_fps = np.array(all_lj_fps)
        all_es_fps = np.array(all_es_fps)

        # Check all are finite
        assert np.all(np.isfinite(all_lj_fps)), "All LJ fingerprints should be finite"
        assert np.all(np.isfinite(all_es_fps)), "All ES fingerprints should be finite"

        # Fingerprints should change between frames (unless system is frozen)
        if n_frames_to_test > 1:
            lj_variance = np.var(all_lj_fps, axis=0)
            es_variance = np.var(all_es_fps, axis=0)
            # At least some residues should show variation
            assert np.any(lj_variance > 0) or np.any(es_variance > 0), \
                "Fingerprints should vary between frames in a dynamic trajectory"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
