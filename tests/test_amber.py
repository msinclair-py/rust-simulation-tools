"""Tests for AMBER file readers (prmtop and inpcrd)."""

import numpy as np
import pytest
from pathlib import Path

from rust_simulation_tools import read_prmtop, read_inpcrd


# Path to test data directory
DATA_DIR = Path(__file__).parent.parent / "data"


class TestReadPrmtop:
    """Test suite for AMBER prmtop file reading."""

    @pytest.fixture
    def prmtop_path(self):
        """Get path to test prmtop file."""
        path = DATA_DIR / "amber.prmtop"
        if not path.exists():
            pytest.skip("amber.prmtop not found in data directory")
        return str(path)

    def test_read_prmtop_basic(self, prmtop_path):
        """Test that prmtop file can be read without errors."""
        topo = read_prmtop(prmtop_path)

        assert topo is not None
        assert topo.n_atoms > 0, "Should have at least one atom"
        assert topo.n_residues > 0, "Should have at least one residue"

    def test_prmtop_atom_count(self, prmtop_path):
        """Test atom count property."""
        topo = read_prmtop(prmtop_path)

        n_atoms = topo.n_atoms
        assert isinstance(n_atoms, int)
        assert n_atoms > 0

        # Verify consistency with other arrays
        charges = topo.charges()
        assert len(charges) == n_atoms

    def test_prmtop_residue_count(self, prmtop_path):
        """Test residue count property."""
        topo = read_prmtop(prmtop_path)

        n_residues = topo.n_residues
        assert isinstance(n_residues, int)
        assert n_residues > 0
        assert n_residues <= topo.n_atoms  # Can't have more residues than atoms

    def test_prmtop_atom_names(self, prmtop_path):
        """Test atom names retrieval."""
        topo = read_prmtop(prmtop_path)

        atom_names = topo.atom_names
        assert len(atom_names) == topo.n_atoms
        assert all(isinstance(name, str) for name in atom_names)

    def test_prmtop_residue_labels(self, prmtop_path):
        """Test residue labels retrieval."""
        topo = read_prmtop(prmtop_path)

        residue_labels = topo.residue_labels
        assert len(residue_labels) == topo.n_residues
        assert all(isinstance(label, str) for label in residue_labels)

    def test_prmtop_charges(self, prmtop_path):
        """Test charge retrieval and unit conversion."""
        topo = read_prmtop(prmtop_path)

        charges = np.array(topo.charges())
        assert len(charges) == topo.n_atoms
        assert charges.dtype == np.float64

        # Charges should be in elementary charge units (typically -2 to +2 range)
        # After conversion from AMBER units
        assert np.all(np.abs(charges) < 10), \
            "Charges should be converted to elementary units (small values)"

    def test_prmtop_sigmas(self, prmtop_path):
        """Test LJ sigma parameter retrieval."""
        topo = read_prmtop(prmtop_path)

        sigmas = np.array(topo.sigmas())
        assert len(sigmas) == topo.n_atoms
        assert sigmas.dtype == np.float64

        # Sigmas should be in nm (typically 0.1-0.5 nm range)
        assert np.all(sigmas >= 0), "Sigmas should be non-negative"
        assert np.all(sigmas < 1.0), "Sigmas should be in nm (less than 1 nm)"

    def test_prmtop_epsilons(self, prmtop_path):
        """Test LJ epsilon parameter retrieval."""
        topo = read_prmtop(prmtop_path)

        epsilons = np.array(topo.epsilons())
        assert len(epsilons) == topo.n_atoms
        assert epsilons.dtype == np.float64

        # Epsilons should be in kJ/mol (typically 0-5 kJ/mol range)
        assert np.all(epsilons >= 0), "Epsilons should be non-negative"

    def test_prmtop_atom_residue_indices(self, prmtop_path):
        """Test atom to residue index mapping."""
        topo = read_prmtop(prmtop_path)

        atom_res_indices = np.array(topo.atom_residue_indices())
        assert len(atom_res_indices) == topo.n_atoms

        # Residue indices should be in range [0, n_residues)
        assert np.all(atom_res_indices >= 0)
        assert np.all(atom_res_indices < topo.n_residues)

        # Should be monotonically non-decreasing (atoms sorted by residue)
        assert np.all(np.diff(atom_res_indices) >= 0), \
            "Atom residue indices should be monotonically non-decreasing"

    def test_prmtop_build_resmap(self, prmtop_path):
        """Test resmap building for fingerprint calculations."""
        topo = read_prmtop(prmtop_path)

        resmap_indices, resmap_offsets = topo.build_resmap()
        resmap_indices = np.array(resmap_indices)
        resmap_offsets = np.array(resmap_offsets)

        # Offsets should have n_residues + 1 entries
        assert len(resmap_offsets) == topo.n_residues + 1

        # First offset should be 0
        assert resmap_offsets[0] == 0

        # Last offset should equal number of atoms
        assert resmap_offsets[-1] == topo.n_atoms

        # Offsets should be monotonically increasing
        assert np.all(np.diff(resmap_offsets) >= 0)

        # Indices should be valid atom indices
        assert np.all(resmap_indices >= 0)
        assert np.all(resmap_indices < topo.n_atoms)

    def test_prmtop_residue_pointers(self, prmtop_path):
        """Test residue pointer retrieval."""
        topo = read_prmtop(prmtop_path)

        residue_ptrs = np.array(topo.residue_pointers())
        assert len(residue_ptrs) == topo.n_residues

        # First residue should start at atom 0
        assert residue_ptrs[0] == 0

        # Pointers should be monotonically increasing
        assert np.all(np.diff(residue_ptrs) >= 0)

        # All pointers should be valid atom indices
        assert np.all(residue_ptrs >= 0)
        assert np.all(residue_ptrs < topo.n_atoms)


class TestReadPrmtopErrors:
    """Test error handling for prmtop reading."""

    def test_nonexistent_file(self):
        """Test that reading nonexistent file raises error."""
        with pytest.raises(Exception):
            read_prmtop("/nonexistent/path/to/file.prmtop")

    def test_invalid_file(self, tmp_path):
        """Test that reading invalid file raises error."""
        # Create a file with invalid content
        invalid_file = tmp_path / "invalid.prmtop"
        invalid_file.write_text("This is not a valid prmtop file")

        with pytest.raises(Exception):
            read_prmtop(str(invalid_file))


class TestReadInpcrd:
    """Test suite for AMBER inpcrd/rst7 file reading."""

    @pytest.fixture
    def inpcrd_content(self, tmp_path):
        """Create a simple test inpcrd file."""
        # Simple 3-atom inpcrd in standard format
        content = """Test inpcrd file
     3
   1.0000000   2.0000000   3.0000000   4.0000000   5.0000000   6.0000000
   7.0000000   8.0000000   9.0000000
"""
        inpcrd_file = tmp_path / "test.inpcrd"
        inpcrd_file.write_text(content)
        return str(inpcrd_file)

    @pytest.fixture
    def inpcrd_with_box(self, tmp_path):
        """Create a test inpcrd file with box dimensions."""
        # 2-atom inpcrd with box
        content = """Test inpcrd with box
     2
   1.0000000   2.0000000   3.0000000   4.0000000   5.0000000   6.0000000
  50.0000000  50.0000000  50.0000000  90.0000000  90.0000000  90.0000000
"""
        inpcrd_file = tmp_path / "test_box.inpcrd"
        inpcrd_file.write_text(content)
        return str(inpcrd_file)

    def test_read_inpcrd_basic(self, inpcrd_content):
        """Test basic inpcrd file reading."""
        positions, box_dims = read_inpcrd(inpcrd_content)

        assert positions is not None
        assert positions.shape == (3, 3)  # 3 atoms, 3 coordinates each

    def test_read_inpcrd_coordinates(self, inpcrd_content):
        """Test that coordinates are read correctly and converted to nm."""
        positions, box_dims = read_inpcrd(inpcrd_content)

        # Original coordinates in Angstrom: 1,2,3; 4,5,6; 7,8,9
        # Should be converted to nm (divide by 10)
        expected = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ])

        np.testing.assert_allclose(positions, expected, rtol=1e-5)

    def test_read_inpcrd_no_box(self, inpcrd_content):
        """Test that file without box returns None for box dimensions."""
        positions, box_dims = read_inpcrd(inpcrd_content)
        assert box_dims is None

    def test_read_inpcrd_with_box(self, inpcrd_with_box):
        """Test reading inpcrd file with box dimensions."""
        positions, box_dims = read_inpcrd(inpcrd_with_box)

        assert positions.shape == (2, 3)
        assert box_dims is not None
        assert len(box_dims) == 3

        # Box should be 50 Angstrom = 5.0 nm
        expected_box = [5.0, 5.0, 5.0]
        np.testing.assert_allclose(box_dims, expected_box, rtol=1e-5)


class TestReadInpcrdErrors:
    """Test error handling for inpcrd reading."""

    def test_nonexistent_file(self):
        """Test that reading nonexistent file raises error."""
        with pytest.raises(Exception):
            read_inpcrd("/nonexistent/path/to/file.inpcrd")

    def test_empty_file(self, tmp_path):
        """Test that reading empty file raises error."""
        empty_file = tmp_path / "empty.inpcrd"
        empty_file.write_text("")

        with pytest.raises(Exception):
            read_inpcrd(str(empty_file))


class TestIntegration:
    """Integration tests using real data files."""

    @pytest.fixture
    def real_prmtop(self):
        """Get path to real prmtop file if available."""
        path = DATA_DIR / "amber.prmtop"
        if not path.exists():
            pytest.skip("amber.prmtop not found")
        return str(path)

    def test_topology_consistency(self, real_prmtop):
        """Test that all topology data is internally consistent."""
        topo = read_prmtop(real_prmtop)

        # All arrays should have n_atoms length
        n_atoms = topo.n_atoms
        assert len(topo.atom_names) == n_atoms
        assert len(np.array(topo.charges())) == n_atoms
        assert len(np.array(topo.sigmas())) == n_atoms
        assert len(np.array(topo.epsilons())) == n_atoms
        assert len(np.array(topo.atom_residue_indices())) == n_atoms

        # Residue arrays should have n_residues length
        n_residues = topo.n_residues
        assert len(topo.residue_labels) == n_residues
        assert len(np.array(topo.residue_pointers())) == n_residues

        # Resmap offsets should have n_residues + 1 length
        _, offsets = topo.build_resmap()
        assert len(offsets) == n_residues + 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
