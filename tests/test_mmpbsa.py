"""Tests for MM-PBSA binding free energy calculations."""

import numpy as np
import pytest
from pathlib import Path

from rust_simulation_tools import (
    read_prmtop,
    GbModel,
    GbParams,
    SaParams,
    DcdReader,
    compute_mm_energy,
    compute_gb_energy,
    compute_sa_energy,
    compute_binding_energy,
    compute_binding_energy_single_frame,
    decompose_binding_energy,
    interaction_entropy,
    quasi_harmonic_entropy,
    extract_subtopology,
)


DATA_DIR = Path(__file__).parent.parent / "data"
# Number of residues to extract for fast tests
N_TEST_RESIDUES = 20


@pytest.fixture
def full_topology():
    """Load the full AMBER topology."""
    path = DATA_DIR / "amber.prmtop"
    if not path.exists():
        pytest.skip("amber.prmtop not found in data directory")
    return read_prmtop(str(path))


@pytest.fixture
def dcd_path():
    """Path to the full DCD trajectory (matches amber.prmtop)."""
    path = DATA_DIR / "amber.dcd"
    if not path.exists():
        pytest.skip("amber.dcd not found in data directory")
    return str(path)


@pytest.fixture
def first_frame_angstrom(full_topology, dcd_path):
    """First frame coordinates in Angstroms (DcdReader returns nm)."""
    reader = DcdReader(dcd_path)
    frame_nm, _box = reader.read_frame()
    return frame_nm * 10.0


@pytest.fixture
def sub_topology(full_topology):
    """Extract a small sub-topology (first N_TEST_RESIDUES residues) for fast tests."""
    ptrs = np.array(full_topology.residue_pointers())
    end_atom = int(ptrs[N_TEST_RESIDUES])
    atom_indices = list(range(end_atom))
    return extract_subtopology(full_topology, atom_indices), atom_indices


@pytest.fixture
def sub_coords(first_frame_angstrom, sub_topology):
    """Coordinates for the sub-topology atoms in Angstroms."""
    _sub_topo, atom_indices = sub_topology
    return first_frame_angstrom[atom_indices]


@pytest.fixture
def sub_topo(sub_topology):
    """Just the sub-topology object."""
    return sub_topology[0]


@pytest.fixture
def residue_split():
    """Split the test residues into receptor and ligand halves."""
    mid = N_TEST_RESIDUES // 2
    return list(range(0, mid)), list(range(mid, N_TEST_RESIDUES))


class TestMmEnergy:
    """Tests for molecular mechanics energy calculations."""

    def test_compute_mm_energy(self, sub_topo, sub_coords):
        """MM energy returns valid result with all components."""
        mm = compute_mm_energy(sub_topo, sub_coords)

        assert mm is not None
        assert isinstance(mm.bond, float)
        assert isinstance(mm.angle, float)
        assert isinstance(mm.dihedral, float)
        assert isinstance(mm.vdw, float)
        assert isinstance(mm.elec, float)
        assert isinstance(mm.vdw_14, float)
        assert isinstance(mm.elec_14, float)

    def test_mm_energy_total(self, sub_topo, sub_coords):
        """Total MM energy equals the sum of components."""
        mm = compute_mm_energy(sub_topo, sub_coords)

        expected_total = (
            mm.bond + mm.angle + mm.dihedral
            + mm.vdw + mm.elec + mm.vdw_14 + mm.elec_14
        )
        assert abs(mm.total() - expected_total) < 1e-6

    def test_mm_energy_bond_nonnegative(self, sub_topo, sub_coords):
        """Bond energy should be non-negative (harmonic potential)."""
        mm = compute_mm_energy(sub_topo, sub_coords)
        assert mm.bond >= 0.0

    def test_mm_energy_angle_nonnegative(self, sub_topo, sub_coords):
        """Angle energy should be non-negative (harmonic potential)."""
        mm = compute_mm_energy(sub_topo, sub_coords)
        assert mm.angle >= 0.0

    def test_mm_energy_reasonable_magnitude(self, sub_topo, sub_coords):
        """MM energy components should be in a physically reasonable range."""
        mm = compute_mm_energy(sub_topo, sub_coords)
        # For a small protein fragment, individual terms should not be astronomical
        for component in [mm.bond, mm.angle, mm.dihedral, mm.vdw, mm.elec]:
            assert np.isfinite(component)
            assert abs(component) < 1e8


class TestGbEnergy:
    """Tests for Generalized Born solvation energy."""

    def test_compute_gb_energy_default(self, sub_topo, sub_coords):
        """GB energy with default parameters returns valid result."""
        gb = compute_gb_energy(sub_topo, sub_coords)

        assert gb is not None
        assert isinstance(gb.total, float)
        assert gb.total < 0.0, "GB solvation energy should be negative"

    def test_compute_gb_energy_obc2(self, sub_topo, sub_coords):
        """GB energy with OBC-II model."""
        params = GbParams(model=GbModel.ObcII, salt_concentration=0.15)
        gb = compute_gb_energy(sub_topo, sub_coords, params)

        assert gb.total < 0.0

    def test_gb_born_radii(self, sub_topo, sub_coords):
        """Born radii array has correct length and positive values."""
        gb = compute_gb_energy(sub_topo, sub_coords)
        radii = np.array(gb.born_radii())

        assert len(radii) == sub_topo.n_atoms
        assert np.all(radii > 0.0), "Born radii should be positive"

    @pytest.mark.parametrize("model", [GbModel.Hct, GbModel.ObcI, GbModel.ObcII])
    def test_gb_all_models(self, sub_topo, sub_coords, model):
        """All GB models produce finite negative solvation energy."""
        params = GbParams(model=model)
        gb = compute_gb_energy(sub_topo, sub_coords, params)

        assert np.isfinite(gb.total)
        assert gb.total < 0.0


class TestSaEnergy:
    """Tests for surface area (non-polar solvation) energy."""

    def test_compute_sa_energy_default(self, sub_topo, sub_coords):
        """SA energy with default parameters."""
        sa = compute_sa_energy(sub_topo, sub_coords)

        assert sa is not None
        assert isinstance(sa.total, float)
        assert sa.total_sasa > 0.0, "Total SASA should be positive"

    def test_compute_sa_energy_custom(self, sub_topo, sub_coords):
        """SA energy with custom surface tension and probe radius."""
        params = SaParams(probe_radius=1.4, surface_tension=0.0072)
        sa = compute_sa_energy(sub_topo, sub_coords, params)

        assert sa.total_sasa > 0.0
        assert np.isfinite(sa.total)

    def test_sa_per_atom_sasa(self, sub_topo, sub_coords):
        """Per-atom SASA array has correct length and non-negative values."""
        sa = compute_sa_energy(sub_topo, sub_coords)
        per_atom = np.array(sa.per_atom_sasa())

        assert len(per_atom) == sub_topo.n_atoms
        assert np.all(per_atom >= 0.0), "Per-atom SASA should be non-negative"

    def test_sa_total_is_sum_of_per_atom(self, sub_topo, sub_coords):
        """Total SASA should equal sum of per-atom SASA values."""
        sa = compute_sa_energy(sub_topo, sub_coords)
        per_atom = np.array(sa.per_atom_sasa())

        np.testing.assert_allclose(sa.total_sasa, per_atom.sum(), rtol=1e-5)


class TestBindingEnergySingleFrame:
    """Tests for single-frame binding free energy."""

    def test_single_frame_binding(self, sub_topo, sub_coords, residue_split):
        """Single-frame binding energy produces valid delta values."""
        receptor_res, ligand_res = residue_split

        frame = compute_binding_energy_single_frame(
            sub_topo, sub_coords,
            receptor_residues=receptor_res,
            ligand_residues=ligand_res,
        )

        assert frame is not None
        assert np.isfinite(frame.delta_mm)
        assert np.isfinite(frame.delta_gb)
        assert np.isfinite(frame.delta_sa)
        assert np.isfinite(frame.delta_total)

    def test_single_frame_delta_decomposition(self, sub_topo, sub_coords, residue_split):
        """Delta total should equal sum of delta components."""
        receptor_res, ligand_res = residue_split

        frame = compute_binding_energy_single_frame(
            sub_topo, sub_coords,
            receptor_residues=receptor_res,
            ligand_residues=ligand_res,
        )

        expected = frame.delta_mm + frame.delta_gb + frame.delta_sa
        assert abs(frame.delta_total - expected) < 1e-4

    def test_single_frame_with_params(self, sub_topo, sub_coords, residue_split):
        """Single-frame binding with explicit GB/SA parameters."""
        receptor_res, ligand_res = residue_split

        frame = compute_binding_energy_single_frame(
            sub_topo, sub_coords,
            receptor_residues=receptor_res,
            ligand_residues=ligand_res,
            gb_params=GbParams(model=GbModel.ObcII, salt_concentration=0.15),
            sa_params=SaParams(probe_radius=1.4, surface_tension=0.0072),
        )

        assert np.isfinite(frame.delta_total)

    def test_single_frame_subsystem_energies(self, sub_topo, sub_coords, residue_split):
        """Complex, receptor, and ligand energies are all finite."""
        receptor_res, ligand_res = residue_split

        frame = compute_binding_energy_single_frame(
            sub_topo, sub_coords,
            receptor_residues=receptor_res,
            ligand_residues=ligand_res,
        )

        for attr in [
            "complex_mm", "complex_gb", "complex_sa", "complex_total",
            "receptor_mm", "receptor_gb", "receptor_sa", "receptor_total",
            "ligand_mm", "ligand_gb", "ligand_sa", "ligand_total",
        ]:
            assert np.isfinite(getattr(frame, attr)), f"{attr} is not finite"

    def test_delta_is_complex_minus_parts(self, sub_topo, sub_coords, residue_split):
        """Delta = complex - receptor - ligand for each component."""
        receptor_res, ligand_res = residue_split

        f = compute_binding_energy_single_frame(
            sub_topo, sub_coords,
            receptor_residues=receptor_res,
            ligand_residues=ligand_res,
        )

        np.testing.assert_allclose(f.delta_mm, f.complex_mm - f.receptor_mm - f.ligand_mm, atol=1e-4)
        np.testing.assert_allclose(f.delta_gb, f.complex_gb - f.receptor_gb - f.ligand_gb, atol=1e-4)
        np.testing.assert_allclose(f.delta_sa, f.complex_sa - f.receptor_sa - f.ligand_sa, atol=1e-4)
        np.testing.assert_allclose(
            f.delta_total, f.complex_total - f.receptor_total - f.ligand_total, atol=1e-4
        )


class TestDecomposition:
    """Tests for per-residue energy decomposition."""

    def test_decompose_binding(self, sub_topo, sub_coords, residue_split):
        """Decomposition produces correct number of residue contributions."""
        receptor_res, ligand_res = residue_split

        decomp = decompose_binding_energy(
            sub_topo, sub_coords,
            receptor_residues=receptor_res,
            ligand_residues=ligand_res,
        )

        assert decomp is not None
        assert len(decomp.receptor_residues) == len(receptor_res)
        assert len(decomp.ligand_residues) == len(ligand_res)

    def test_residue_contribution_fields(self, sub_topo, sub_coords, residue_split):
        """Each residue contribution has expected fields and total = sum of parts."""
        receptor_res, ligand_res = residue_split

        decomp = decompose_binding_energy(
            sub_topo, sub_coords,
            receptor_residues=receptor_res,
            ligand_residues=ligand_res,
        )

        for res in decomp.receptor_residues:
            assert isinstance(res.residue_index, int)
            assert isinstance(res.residue_label, str)
            assert len(res.residue_label) > 0
            assert np.isfinite(res.vdw)
            assert np.isfinite(res.elec)
            assert np.isfinite(res.gb)
            assert np.isfinite(res.sa)
            expected_total = res.vdw + res.elec + res.gb + res.sa
            assert abs(res.total() - expected_total) < 1e-6


class TestTrajectoryBinding:
    """Tests for multi-frame trajectory binding energy.

    These tests use the full topology + DCD and may be slow.
    """

    @pytest.mark.slow
    def test_trajectory_binding_dcd(self, full_topology, dcd_path):
        """Trajectory binding energy from DCD file with stride to limit frames."""
        n_res = full_topology.n_residues
        mid = n_res // 2
        receptor_res = list(range(0, mid))
        ligand_res = list(range(mid, n_res))

        result = compute_binding_energy(
            full_topology,
            trajectory_path=dcd_path,
            receptor_residues=receptor_res,
            ligand_residues=ligand_res,
            trajectory_format="dcd",
            stride=500,  # large stride to only process a few frames
        )

        assert result is not None
        assert len(result.frames) > 0
        assert np.isfinite(result.mean_delta_total)
        assert np.isfinite(result.std_delta_total)
        assert result.sem_delta_total >= 0.0

    @pytest.mark.slow
    def test_trajectory_last_frame_coords(self, full_topology, dcd_path):
        """Last frame coordinates have the correct shape."""
        n_res = full_topology.n_residues
        mid = n_res // 2

        result = compute_binding_energy(
            full_topology,
            trajectory_path=dcd_path,
            receptor_residues=list(range(0, mid)),
            ligand_residues=list(range(mid, n_res)),
            trajectory_format="dcd",
            stride=500,
        )

        last_coords = result.last_frame_coords()
        assert last_coords.shape == (full_topology.n_atoms, 3)


class TestEntropy:
    """Tests for entropy estimation methods."""

    def test_quasi_harmonic_entropy(self, sub_topo, sub_coords):
        """Quasi-harmonic entropy from a synthetic multi-frame trajectory."""
        n_atoms = sub_topo.n_atoms
        n_frames = 50
        # Create a synthetic trajectory with small perturbations
        rng = np.random.default_rng(42)
        traj = np.stack([sub_coords + rng.normal(0, 0.1, sub_coords.shape) for _ in range(n_frames)])

        masses = np.ones(n_atoms) * 12.0

        qh = quasi_harmonic_entropy(traj, masses, temperature=298.15)
        if qh is not None:
            assert np.isfinite(qh.minus_tds)
            assert qh.method == "quasi_harmonic"


class TestSubtopology:
    """Tests for sub-topology extraction."""

    def test_extract_subtopology(self, full_topology):
        """Extracting a subset of atoms produces a valid topology."""
        n_subset = min(100, full_topology.n_atoms)
        atom_indices = list(range(n_subset))
        sub = extract_subtopology(full_topology, atom_indices)

        assert sub.n_atoms == n_subset
        assert sub.n_residues > 0
        assert sub.n_residues <= full_topology.n_residues

    def test_subtopology_consistency(self, sub_topo):
        """Sub-topology has consistent internal data."""
        assert sub_topo.n_atoms > 0
        assert sub_topo.n_residues == N_TEST_RESIDUES
        assert len(sub_topo.atom_names) == sub_topo.n_atoms
        assert len(sub_topo.residue_labels) == sub_topo.n_residues
        charges = np.array(sub_topo.charges())
        assert len(charges) == sub_topo.n_atoms

    def test_subtopology_energy(self, sub_topo, sub_coords):
        """MM energy can be computed on an extracted sub-topology."""
        mm = compute_mm_energy(sub_topo, sub_coords)
        assert np.isfinite(mm.total())


class TestSolvatedTopology:
    """Tests that MM-PBSA works transparently with solvated topologies."""

    def test_solvated_matches_stripped(self, full_topology, first_frame_angstrom):
        """Binding energy from solvated topology matches stripped-topology result.

        Passes the full solvated topology + receptor/ligand residue indices and
        verifies the result matches computing on a pre-stripped complex sub-topology.
        """
        # Use first N_TEST_RESIDUES residues, split into receptor/ligand halves.
        mid = N_TEST_RESIDUES // 2
        receptor_res_orig = list(range(0, mid))
        ligand_res_orig = list(range(mid, N_TEST_RESIDUES))

        # --- Stripped path: manually extract complex sub-topology first ---
        complex_res = list(range(0, N_TEST_RESIDUES))
        complex_sel = full_topology.build_selection(complex_res)
        complex_atom_indices = list(int(x) for x in complex_sel["atom_indices"])
        stripped_top = extract_subtopology(full_topology, complex_atom_indices)
        stripped_coords = first_frame_angstrom[complex_atom_indices]

        # Residues in stripped topology are 0..N_TEST_RESIDUES (identity mapping)
        stripped_receptor_res = list(range(0, mid))
        stripped_ligand_res = list(range(mid, N_TEST_RESIDUES))

        frame_stripped = compute_binding_energy_single_frame(
            stripped_top, stripped_coords,
            receptor_residues=stripped_receptor_res,
            ligand_residues=stripped_ligand_res,
        )

        # --- Solvated path: pass solvated topology directly ---
        frame_solvated = compute_binding_energy_single_frame(
            full_topology, first_frame_angstrom,
            receptor_residues=receptor_res_orig,
            ligand_residues=ligand_res_orig,
        )

        # Results should match exactly (same atoms, same computation).
        np.testing.assert_allclose(frame_solvated.delta_total, frame_stripped.delta_total, atol=1e-4)
        np.testing.assert_allclose(frame_solvated.delta_mm, frame_stripped.delta_mm, atol=1e-4)
        np.testing.assert_allclose(frame_solvated.delta_gb, frame_stripped.delta_gb, atol=1e-4)
        np.testing.assert_allclose(frame_solvated.delta_sa, frame_stripped.delta_sa, atol=1e-4)


class TestGbParamsConfig:
    """Tests for GB parameter configuration."""

    def test_default_params(self, sub_topo, sub_coords):
        """Default GbParams produce valid energy."""
        params = GbParams()
        gb = compute_gb_energy(sub_topo, sub_coords, params)
        assert np.isfinite(gb.total)

    def test_salt_concentration(self, sub_topo, sub_coords):
        """Different salt concentrations produce different energies."""
        gb_0 = compute_gb_energy(sub_topo, sub_coords, GbParams(salt_concentration=0.0))
        gb_1 = compute_gb_energy(sub_topo, sub_coords, GbParams(salt_concentration=0.15))

        assert gb_0.total != gb_1.total

    def test_different_models_different_energies(self, sub_topo, sub_coords):
        """Different GB models produce different solvation energies."""
        gb_hct = compute_gb_energy(sub_topo, sub_coords, GbParams(model=GbModel.Hct))
        gb_obc2 = compute_gb_energy(sub_topo, sub_coords, GbParams(model=GbModel.ObcII))

        assert gb_hct.total != gb_obc2.total


class TestSaParamsConfig:
    """Tests for SA parameter configuration."""

    def test_default_params(self, sub_topo, sub_coords):
        """Default SaParams produce valid energy."""
        params = SaParams()
        sa = compute_sa_energy(sub_topo, sub_coords, params)
        assert np.isfinite(sa.total)
        assert sa.total_sasa > 0.0

    def test_different_surface_tension(self, sub_topo, sub_coords):
        """Different surface tension values produce proportionally different energies."""
        sa_low = compute_sa_energy(sub_topo, sub_coords, SaParams(surface_tension=0.005))
        sa_high = compute_sa_energy(sub_topo, sub_coords, SaParams(surface_tension=0.01))

        # Same SASA, different gamma
        assert sa_low.total != sa_high.total
        assert abs(sa_high.total) > abs(sa_low.total)

    def test_different_probe_radius(self, sub_topo, sub_coords):
        """Different probe radii produce different SASA values."""
        sa_small = compute_sa_energy(sub_topo, sub_coords, SaParams(probe_radius=1.0))
        sa_large = compute_sa_energy(sub_topo, sub_coords, SaParams(probe_radius=2.0))

        assert sa_large.total_sasa != sa_small.total_sasa


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
