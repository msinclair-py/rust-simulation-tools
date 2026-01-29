"""Tests for FingerprintSession high-level API."""

import numpy as np
import pytest
from pathlib import Path

from rust_simulation_tools import (
    FingerprintSession,
    FingerprintMode,
    compute_fingerprints,
    read_prmtop,
    DcdReader,
)


# Path to test data directory
DATA_DIR = Path(__file__).parent.parent / "data"


class TestFingerprintSessionBasic:
    """Basic tests for FingerprintSession initialization and attributes."""

    @pytest.fixture
    def amber_data_paths(self):
        """Return paths to AMBER test data if available."""
        prmtop_path = DATA_DIR / "amber.prmtop"
        dcd_path = DATA_DIR / "amber.dcd"
        if not prmtop_path.exists():
            pytest.skip("amber.prmtop not found in data directory")
        if not dcd_path.exists():
            pytest.skip("amber.dcd not found in data directory")
        return str(prmtop_path), str(dcd_path)

    def test_session_creation_with_topology_only(self, amber_data_paths):
        """Test creating session with topology only."""
        prmtop_path, _ = amber_data_paths
        session = FingerprintSession(prmtop_path)

        assert session.n_atoms > 0
        assert session.n_residues > 0
        assert session.n_frames == 0  # No DCD loaded

    def test_session_creation_with_trajectory(self, amber_data_paths):
        """Test creating session with topology and trajectory."""
        prmtop_path, dcd_path = amber_data_paths
        session = FingerprintSession(prmtop_path, dcd_path)

        assert session.n_atoms > 0
        assert session.n_residues > 0
        assert session.n_frames > 0
        assert session.current_frame == 0

    def test_set_target_residues(self, amber_data_paths):
        """Test setting target residue selection."""
        prmtop_path, dcd_path = amber_data_paths
        session = FingerprintSession(prmtop_path, dcd_path)

        # Set first 10 residues as target
        n_target = min(10, session.n_residues // 2)
        session.set_target_residues(list(range(n_target)))

    def test_set_binder_residues(self, amber_data_paths):
        """Test setting binder residue selection."""
        prmtop_path, dcd_path = amber_data_paths
        session = FingerprintSession(prmtop_path, dcd_path)

        # Set last 10 residues as binder
        n_binder = min(10, session.n_residues // 2)
        start = session.n_residues - n_binder
        session.set_binder_residues(list(range(start, session.n_residues)))

    def test_invalid_residue_index(self, amber_data_paths):
        """Test that invalid residue indices raise errors."""
        prmtop_path, dcd_path = amber_data_paths
        session = FingerprintSession(prmtop_path, dcd_path)

        with pytest.raises(ValueError, match="out of range"):
            session.set_target_residues([session.n_residues + 100])

    def test_overlapping_selections_error(self, amber_data_paths):
        """Test that overlapping target/binder selections raise errors."""
        prmtop_path, dcd_path = amber_data_paths
        session = FingerprintSession(prmtop_path, dcd_path)

        session.set_target_residues([0, 1, 2, 3, 4])

        with pytest.raises(ValueError, match="overlap"):
            session.set_binder_residues([3, 4, 5, 6, 7])  # Overlaps with target

    def test_fingerprint_mode_setting(self, amber_data_paths):
        """Test setting fingerprint mode."""
        prmtop_path, dcd_path = amber_data_paths
        session = FingerprintSession(prmtop_path, dcd_path)

        # Default is Target mode
        assert session.fingerprint_mode == FingerprintMode.Target

        # Set to Binder mode
        session.set_fingerprint_mode(FingerprintMode.Binder)
        assert session.fingerprint_mode == FingerprintMode.Binder

        # Set back to Target mode
        session.set_fingerprint_mode(FingerprintMode.Target)
        assert session.fingerprint_mode == FingerprintMode.Target


class TestFingerprintSessionComputation:
    """Test fingerprint computation with FingerprintSession."""

    @pytest.fixture
    def configured_session(self):
        """Return a configured FingerprintSession if test data available."""
        prmtop_path = DATA_DIR / "amber.prmtop"
        dcd_path = DATA_DIR / "amber.dcd"
        if not prmtop_path.exists() or not dcd_path.exists():
            pytest.skip("AMBER test data not found")

        session = FingerprintSession(str(prmtop_path), str(dcd_path))

        # Use first 10 residues as target, rest as binder
        n_target = min(10, session.n_residues // 2)
        n_binder = session.n_residues - n_target
        if n_binder < 1:
            pytest.skip("Not enough residues for target/binder split")

        session.set_target_residues(list(range(n_target)))
        session.set_binder_residues(list(range(n_target, session.n_residues)))

        return session, n_target, n_binder

    def test_compute_next_frame(self, configured_session):
        """Test computing fingerprints for a single frame."""
        session, n_target, _ = configured_session

        result = session.compute_next_frame()
        assert result is not None

        lj_fp, es_fp = result
        assert len(lj_fp) == n_target
        assert len(es_fp) == n_target
        assert np.all(np.isfinite(lj_fp))
        assert np.all(np.isfinite(es_fp))

    def test_iteration(self, configured_session):
        """Test iterating over trajectory frames."""
        session, n_target, _ = configured_session

        frame_count = 0
        for lj_fp, es_fp in session:
            assert len(lj_fp) == n_target
            assert len(es_fp) == n_target
            assert np.all(np.isfinite(lj_fp))
            assert np.all(np.isfinite(es_fp))
            frame_count += 1
            if frame_count >= 3:  # Only test first few frames
                break

        assert frame_count > 0

    def test_seek_functionality(self, configured_session):
        """Test seeking to specific frames."""
        session, n_target, _ = configured_session

        # Read first frame
        session.seek(0)
        result1 = session.compute_next_frame()
        assert result1 is not None
        lj_fp1, es_fp1 = result1

        # Read again after seeking back
        session.seek(0)
        result2 = session.compute_next_frame()
        assert result2 is not None
        lj_fp2, es_fp2 = result2

        # Should be identical
        np.testing.assert_array_almost_equal(lj_fp1, lj_fp2)
        np.testing.assert_array_almost_equal(es_fp1, es_fp2)

    def test_residue_labels(self, configured_session):
        """Test that residue labels are accessible."""
        session, n_target, _ = configured_session

        # In Target mode, should return target residue labels
        labels = session.residue_labels
        assert len(labels) == n_target
        assert all(isinstance(label, str) for label in labels)

        # Switch to Binder mode
        session.set_fingerprint_mode(FingerprintMode.Binder)
        binder_labels = session.residue_labels
        assert len(binder_labels) == session.n_residues - n_target


class TestReturnResidueNames:
    """Test the return_residue_names feature."""

    @pytest.fixture
    def configured_session(self):
        """Return a configured FingerprintSession if test data available."""
        prmtop_path = DATA_DIR / "amber.prmtop"
        dcd_path = DATA_DIR / "amber.dcd"
        if not prmtop_path.exists() or not dcd_path.exists():
            pytest.skip("AMBER test data not found")

        session = FingerprintSession(str(prmtop_path), str(dcd_path))

        n_target = min(10, session.n_residues // 2)
        n_binder = session.n_residues - n_target
        if n_binder < 1:
            pytest.skip("Not enough residues for target/binder split")

        session.set_target_residues(list(range(n_target)))
        session.set_binder_residues(list(range(n_target, session.n_residues)))

        return session, n_target, n_binder

    def test_default_is_false(self, configured_session):
        """Test that return_residue_names defaults to False."""
        session, _, _ = configured_session
        assert session.return_residue_names is False

    def test_compute_next_frame_returns_names(self, configured_session):
        """Test that compute_next_frame returns 3-tuple when enabled."""
        session, n_target, _ = configured_session
        session.return_residue_names = True

        result = session.compute_next_frame()
        assert result is not None
        lj_fp, es_fp, resnames = result

        assert len(lj_fp) == n_target
        assert len(es_fp) == n_target
        assert len(resnames) == n_target
        assert all(isinstance(name, str) for name in resnames)

    def test_iteration_returns_names(self, configured_session):
        """Test that iteration yields 3-tuples when enabled."""
        session, n_target, _ = configured_session
        session.return_residue_names = True

        for lj_fp, es_fp, resnames in session:
            assert len(lj_fp) == n_target
            assert len(es_fp) == n_target
            assert len(resnames) == n_target
            assert all(isinstance(name, str) for name in resnames)
            break  # Only test first frame

    def test_binder_mode_returns_binder_names(self, configured_session):
        """Test that binder mode returns binder residue names."""
        session, _, n_binder = configured_session
        session.return_residue_names = True
        session.set_fingerprint_mode(FingerprintMode.Binder)

        result = session.compute_next_frame()
        assert result is not None
        lj_fp, es_fp, resnames = result

        assert len(resnames) == n_binder
        assert all(isinstance(name, str) for name in resnames)

    def test_names_match_residue_labels(self, configured_session):
        """Test that returned names match session.residue_labels."""
        session, n_target, _ = configured_session
        session.return_residue_names = True

        _, _, resnames = session.compute_next_frame()
        expected = session.residue_labels

        assert list(resnames) == list(expected)


class TestFingerprintModeComparison:
    """Test that Target and Binder modes produce correct results."""

    @pytest.fixture
    def session_with_data(self):
        """Return session and comparison data."""
        prmtop_path = DATA_DIR / "amber.prmtop"
        dcd_path = DATA_DIR / "amber.dcd"
        if not prmtop_path.exists() or not dcd_path.exists():
            pytest.skip("AMBER test data not found")

        session = FingerprintSession(str(prmtop_path), str(dcd_path))
        topo = read_prmtop(str(prmtop_path))

        # Use first 10 residues as target, rest as binder
        n_target = min(10, session.n_residues // 2)
        target_residues = list(range(n_target))
        binder_residues = list(range(n_target, session.n_residues))

        session.set_target_residues(target_residues)
        session.set_binder_residues(binder_residues)

        return session, topo, target_residues, binder_residues

    def test_target_mode_matches_existing_api(self, session_with_data):
        """Test that Target mode produces same results as compute_fingerprints()."""
        session, topo, target_residues, binder_residues = session_with_data

        # Get parameters for existing API
        charges = np.array(topo.charges())
        sigmas = np.array(topo.sigmas())
        epsilons = np.array(topo.epsilons())
        resmap_indices, resmap_offsets = topo.build_resmap()
        resmap_indices = np.array(resmap_indices)
        resmap_offsets = np.array(resmap_offsets)

        # Compute target atom indices for existing API
        n_target = len(target_residues)
        target_atom_end = int(resmap_offsets[n_target])
        binder_indices = np.arange(target_atom_end, topo.n_atoms, dtype=np.int64)
        target_resmap_indices = resmap_indices[:target_atom_end]
        target_resmap_offsets = resmap_offsets[: n_target + 1]

        # Read first frame with DcdReader for comparison
        reader = DcdReader(str(DATA_DIR / "amber.dcd"))
        positions, _ = reader.read_frame()

        # Compute with existing API
        lj_existing, es_existing = compute_fingerprints(
            positions,
            charges,
            sigmas,
            epsilons,
            target_resmap_indices,
            target_resmap_offsets,
            binder_indices,
        )

        # Compute with FingerprintSession (Target mode by default)
        session.seek(0)
        lj_session, es_session = session.compute_next_frame()

        # Should match
        np.testing.assert_array_almost_equal(
            lj_session, lj_existing, decimal=10, err_msg="LJ fingerprints should match"
        )
        np.testing.assert_array_almost_equal(
            es_session, es_existing, decimal=10, err_msg="ES fingerprints should match"
        )

    def test_energy_symmetry(self, session_with_data):
        """Test that sum of target and binder fingerprints are equal.

        Due to energy symmetry (E_AB = E_BA), the total interaction energy
        should be identical regardless of which selection we fingerprint.
        """
        session, _, _, _ = session_with_data

        # Compute in Target mode
        session.set_fingerprint_mode(FingerprintMode.Target)
        session.seek(0)
        lj_target, es_target = session.compute_next_frame()
        target_lj_sum = np.sum(lj_target)
        target_es_sum = np.sum(es_target)

        # Compute in Binder mode
        session.set_fingerprint_mode(FingerprintMode.Binder)
        session.seek(0)
        lj_binder, es_binder = session.compute_next_frame()
        binder_lj_sum = np.sum(lj_binder)
        binder_es_sum = np.sum(es_binder)

        # Total energies should be equal (symmetry)
        np.testing.assert_almost_equal(
            target_lj_sum,
            binder_lj_sum,
            decimal=6,
            err_msg="Total LJ energy should be symmetric",
        )
        np.testing.assert_almost_equal(
            target_es_sum,
            binder_es_sum,
            decimal=6,
            err_msg="Total ES energy should be symmetric",
        )

    def test_binder_mode_correct_length(self, session_with_data):
        """Test that Binder mode returns correct number of residues."""
        session, _, target_residues, binder_residues = session_with_data

        n_binder = len(binder_residues)

        session.set_fingerprint_mode(FingerprintMode.Binder)
        session.seek(0)
        lj_fp, es_fp = session.compute_next_frame()

        assert len(lj_fp) == n_binder
        assert len(es_fp) == n_binder
        assert session.n_fingerprint_residues == n_binder


class TestFingerprintSessionEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def session(self):
        """Return a basic session if test data available."""
        prmtop_path = DATA_DIR / "amber.prmtop"
        dcd_path = DATA_DIR / "amber.dcd"
        if not prmtop_path.exists() or not dcd_path.exists():
            pytest.skip("AMBER test data not found")
        return FingerprintSession(str(prmtop_path), str(dcd_path))

    def test_compute_without_selections_error(self, session):
        """Test that computing without selections raises error."""
        with pytest.raises(ValueError, match="selection not set"):
            session.compute_next_frame()

    def test_compute_with_only_target_error(self, session):
        """Test that computing with only target selection raises error."""
        session.set_target_residues([0, 1, 2])
        with pytest.raises(ValueError, match="Binder selection not set"):
            session.compute_next_frame()

    def test_compute_with_only_binder_error(self, session):
        """Test that computing with only binder selection raises error."""
        session.set_binder_residues([0, 1, 2])
        with pytest.raises(ValueError, match="Target selection not set"):
            session.compute_next_frame()

    def test_seek_without_dcd_error(self):
        """Test that seeking without DCD raises error."""
        prmtop_path = DATA_DIR / "amber.prmtop"
        if not prmtop_path.exists():
            pytest.skip("amber.prmtop not found")

        session = FingerprintSession(str(prmtop_path))  # No DCD

        with pytest.raises(ValueError, match="No DCD trajectory loaded"):
            session.seek(0)

    def test_single_residue_selections(self):
        """Test with single-residue target and binder selections."""
        prmtop_path = DATA_DIR / "amber.prmtop"
        dcd_path = DATA_DIR / "amber.dcd"
        if not prmtop_path.exists() or not dcd_path.exists():
            pytest.skip("AMBER test data not found")

        session = FingerprintSession(str(prmtop_path), str(dcd_path))
        if session.n_residues < 2:
            pytest.skip("Need at least 2 residues")

        session.set_target_residues([0])
        session.set_binder_residues([1])

        result = session.compute_next_frame()
        assert result is not None

        lj_fp, es_fp = result
        assert len(lj_fp) == 1
        assert len(es_fp) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
