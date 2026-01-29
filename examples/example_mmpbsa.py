"""MM-GBSA binding free energy example.

Demonstrates:
1. Single-frame MM, GB, and SA energy calculations
2. Per-residue energy decomposition
3. Multi-frame binding free energy from a trajectory
4. Entropy estimation (interaction entropy and quasi-harmonic)
5. Reading AMBER mdcrd trajectories with MdcrdReader
6. Extracting sub-topologies
7. Using solvated (unstripped) topologies directly
"""

import numpy as np
from rust_simulation_tools import (
    read_prmtop,
    read_inpcrd,
    GbModel,
    GbParams,
    SaParams,
    MdcrdReader,
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


def example_single_frame():
    """Compute MM, GB, and SA energies for a single structure."""
    print("=== Single-Frame Energy Calculation ===\n")

    topo = read_prmtop("complex.prmtop")
    coords, _ = read_inpcrd("complex.inpcrd")

    print(f"System: {topo.n_atoms} atoms, {topo.n_residues} residues")

    # Molecular mechanics energy (bonds, angles, dihedrals, vdW, electrostatics)
    mm = compute_mm_energy(topo, coords)
    print(f"\nMM Energy Components (kcal/mol):")
    print(f"  Bond:     {mm.bond:12.3f}")
    print(f"  Angle:    {mm.angle:12.3f}")
    print(f"  Dihedral: {mm.dihedral:12.3f}")
    print(f"  vdW:      {mm.vdw:12.3f}")
    print(f"  Elec:     {mm.elec:12.3f}")
    print(f"  vdW 1-4:  {mm.vdw_14:12.3f}")
    print(f"  Elec 1-4: {mm.elec_14:12.3f}")
    print(f"  Total:    {mm.total():12.3f}")

    # Generalized Born solvation energy (polar contribution)
    gb_params = GbParams(model=GbModel.ObcII, salt_concentration=0.15)
    gb = compute_gb_energy(topo, coords, gb_params)
    print(f"\nGB Solvation Energy: {gb.total:.3f} kcal/mol")
    print(f"  Born radii range: {np.min(gb.born_radii()):.3f} - {np.max(gb.born_radii()):.3f} A")

    # Surface area energy (non-polar contribution)
    sa_params = SaParams(probe_radius=1.4, surface_tension=0.0072)
    sa = compute_sa_energy(topo, coords, sa_params)
    print(f"\nSA Non-Polar Energy: {sa.total:.3f} kcal/mol")
    print(f"  Total SASA: {sa.total_sasa:.1f} A^2")
    print(f"  Per-atom SASA range: {np.min(sa.per_atom_sasa()):.1f} - {np.max(sa.per_atom_sasa()):.1f} A^2")


def example_binding_energy_single_frame():
    """Compute binding free energy for a single snapshot."""
    print("\n=== Single-Frame Binding Energy ===\n")

    topo = read_prmtop("complex.prmtop")
    coords, _ = read_inpcrd("complex.inpcrd")

    # Define receptor and ligand by residue index (0-based)
    receptor_residues = list(range(0, 250))       # e.g. protein residues 0-249
    ligand_residues = list(range(250, topo.n_residues))  # remaining = ligand

    print(f"Receptor: residues 0-249 ({len(receptor_residues)} residues)")
    print(f"Ligand:   residues 250-{topo.n_residues - 1} ({len(ligand_residues)} residues)")

    frame = compute_binding_energy_single_frame(
        topo, coords,
        receptor_residues=receptor_residues,
        ligand_residues=ligand_residues,
        gb_params=GbParams(model=GbModel.ObcII),
        sa_params=SaParams(),
    )

    print(f"\nEnergy Component    Complex    Receptor    Ligand    Delta")
    print(f"  MM             {frame.complex_mm:10.2f} {frame.receptor_mm:10.2f} {frame.ligand_mm:10.2f} {frame.delta_mm:10.2f}")
    print(f"  GB             {frame.complex_gb:10.2f} {frame.receptor_gb:10.2f} {frame.ligand_gb:10.2f} {frame.delta_gb:10.2f}")
    print(f"  SA             {frame.complex_sa:10.2f} {frame.receptor_sa:10.2f} {frame.ligand_sa:10.2f} {frame.delta_sa:10.2f}")
    print(f"  Total          {frame.complex_total:10.2f} {frame.receptor_total:10.2f} {frame.ligand_total:10.2f} {frame.delta_total:10.2f}")


def example_trajectory_binding():
    """Compute binding free energy averaged over a trajectory."""
    print("\n=== Trajectory Binding Energy (MM-GBSA) ===\n")

    topo = read_prmtop("complex.prmtop")

    receptor_residues = list(range(0, 250))
    ligand_residues = list(range(250, topo.n_residues))

    # Using a DCD trajectory
    result = compute_binding_energy(
        topo,
        trajectory_path="trajectory.dcd",
        receptor_residues=receptor_residues,
        ligand_residues=ligand_residues,
        gb_params=GbParams(model=GbModel.ObcII),
        sa_params=SaParams(),
        trajectory_format="dcd",
        stride=1,
        start_frame=0,
        end_frame=0,  # 0 = all frames
    )

    print(f"Frames analyzed: {len(result.frames)}")
    print(f"\nMean Binding Energy (kcal/mol):")
    print(f"  Delta MM:    {result.mean_delta_mm:8.2f} +/- {result.std_delta_mm:.2f}")
    print(f"  Delta GB:    {result.mean_delta_gb:8.2f} +/- {result.std_delta_gb:.2f}")
    print(f"  Delta SA:    {result.mean_delta_sa:8.2f} +/- {result.std_delta_sa:.2f}")
    print(f"  Delta Total: {result.mean_delta_total:8.2f} +/- {result.std_delta_total:.2f}")
    print(f"  SEM:         {result.sem_delta_total:.2f}")

    # Per-frame energies for plotting
    deltas = [f.delta_total for f in result.frames]
    print(f"\nPer-frame range: {min(deltas):.2f} to {max(deltas):.2f} kcal/mol")

    # Entropy correction via interaction entropy
    entropy_est = interaction_entropy(result.frames, temperature=298.15)
    if entropy_est is not None:
        print(f"\nEntropy Correction ({entropy_est.method}):")
        print(f"  -TdS = {entropy_est.minus_tds:.2f} kcal/mol")
        print(f"  Corrected dG = {result.mean_delta_total + entropy_est.minus_tds:.2f} kcal/mol")

    # Last frame coordinates (useful for visualization)
    last_coords = result.last_frame_coords()
    print(f"\nLast frame coords shape: {last_coords.shape}")


def example_decomposition():
    """Per-residue energy decomposition."""
    print("\n=== Per-Residue Decomposition ===\n")

    topo = read_prmtop("complex.prmtop")
    coords, _ = read_inpcrd("complex.inpcrd")

    receptor_residues = list(range(0, 250))
    ligand_residues = list(range(250, topo.n_residues))

    decomp = decompose_binding_energy(
        topo, coords,
        receptor_residues=receptor_residues,
        ligand_residues=ligand_residues,
        gb_params=GbParams(model=GbModel.ObcII),
        sa_params=SaParams(),
    )

    # Top contributing receptor residues
    receptor = sorted(decomp.receptor_residues, key=lambda r: r.total())
    print("Top 10 receptor residue contributions (kcal/mol):")
    print(f"  {'Residue':<12} {'vdW':>8} {'Elec':>8} {'GB':>8} {'SA':>8} {'Total':>8}")
    for res in receptor[:10]:
        label = f"{res.residue_label}{res.residue_index}"
        print(f"  {label:<12} {res.vdw:8.2f} {res.elec:8.2f} {res.gb:8.2f} {res.sa:8.2f} {res.total():8.2f}")

    # Top contributing ligand residues
    ligand = sorted(decomp.ligand_residues, key=lambda r: r.total())
    print(f"\nTop 5 ligand residue contributions (kcal/mol):")
    for res in ligand[:5]:
        label = f"{res.residue_label}{res.residue_index}"
        print(f"  {label}: {res.total():.2f}")


def example_mdcrd_reader():
    """Read AMBER ASCII trajectory (mdcrd) frame by frame."""
    print("\n=== MdcrdReader ===\n")

    topo = read_prmtop("complex.prmtop")
    reader = MdcrdReader("trajectory.mdcrd", n_atoms=topo.n_atoms, has_box=False)

    print(f"Reading mdcrd for {topo.n_atoms} atoms")

    frame_count = 0
    for frame_coords in reader:
        frame_count += 1
        if frame_count <= 3:
            print(f"  Frame {frame_count}: shape={frame_coords.shape}, "
                  f"center of mass = {np.mean(frame_coords, axis=0)}")
    print(f"Total frames read: {frame_count}")


def example_subtopology():
    """Extract a sub-topology for a subset of atoms."""
    print("\n=== Sub-Topology Extraction ===\n")

    topo = read_prmtop("complex.prmtop")
    print(f"Full topology: {topo.n_atoms} atoms, {topo.n_residues} residues")

    # Extract just the first 100 atoms
    atom_indices = list(range(100))
    sub_topo = extract_subtopology(topo, atom_indices)
    print(f"Sub-topology:  {sub_topo.n_atoms} atoms, {sub_topo.n_residues} residues")
    print(f"  Residues: {list(sub_topo.residue_labels)}")


def example_solvated_binding():
    """Compute binding energy from a solvated (unstripped) topology.

    When the topology contains solvent/ions beyond the receptor and ligand,
    the MM-PBSA functions automatically extract the complex (receptor + ligand)
    sub-topology and slice frame coordinates accordingly. No manual stripping
    is required.
    """
    print("\n=== Solvated Topology Binding Energy ===\n")

    # Load the full solvated topology (protein + ligand + water + ions)
    topo = read_prmtop("solvated.prmtop")
    coords, _ = read_inpcrd("solvated.inpcrd")

    print(f"Solvated system: {topo.n_atoms} atoms, {topo.n_residues} residues")

    # Specify only the receptor and ligand residues (0-based).
    # Solvent/ion residues are simply omitted — they will be stripped
    # automatically.
    receptor_residues = list(range(0, 250))   # protein
    ligand_residues = list(range(250, 251))    # single ligand residue

    print(f"Receptor: {len(receptor_residues)} residues")
    print(f"Ligand:   {len(ligand_residues)} residues")
    print(f"(Remaining {topo.n_residues - 251} solvent/ion residues will be stripped automatically)")

    # Single-frame binding energy — works identically to the stripped case
    frame = compute_binding_energy_single_frame(
        topo, coords,
        receptor_residues=receptor_residues,
        ligand_residues=ligand_residues,
        gb_params=GbParams(model=GbModel.ObcII),
        sa_params=SaParams(),
    )

    print(f"\nBinding Energy (kcal/mol):")
    print(f"  Delta MM:    {frame.delta_mm:10.2f}")
    print(f"  Delta GB:    {frame.delta_gb:10.2f}")
    print(f"  Delta SA:    {frame.delta_sa:10.2f}")
    print(f"  Delta Total: {frame.delta_total:10.2f}")

    # Trajectory binding energy also works with solvated topologies
    result = compute_binding_energy(
        topo,
        trajectory_path="solvated_trajectory.dcd",
        receptor_residues=receptor_residues,
        ligand_residues=ligand_residues,
        gb_params=GbParams(model=GbModel.ObcII),
        sa_params=SaParams(),
        trajectory_format="dcd",
        stride=10,
    )

    print(f"\nTrajectory ({len(result.frames)} frames):")
    print(f"  Mean Delta Total: {result.mean_delta_total:8.2f} +/- {result.std_delta_total:.2f}")


def example_quasi_harmonic():
    """Quasi-harmonic entropy from a 3D trajectory."""
    print("\n=== Quasi-Harmonic Entropy ===\n")

    topo = read_prmtop("complex.prmtop")

    # Build a (n_frames, n_atoms, 3) trajectory array
    # In practice, load from DcdReader.read_all() or similar
    from rust_simulation_tools import DcdReader
    reader = DcdReader("trajectory.dcd")
    traj, _ = reader.read_all()
    print(f"Trajectory shape: {traj.shape}")

    # Masses from topology (approximate: use atom names)
    masses = np.ones(topo.n_atoms) * 12.0  # placeholder; use real masses

    qh = quasi_harmonic_entropy(traj, masses, temperature=298.15)
    if qh is not None:
        print(f"Quasi-harmonic entropy ({qh.method}):")
        print(f"  -TdS = {qh.minus_tds:.2f} kcal/mol")
    else:
        print("Could not compute quasi-harmonic entropy (not enough frames?)")


if __name__ == "__main__":
    import sys

    examples = {
        "single":       example_single_frame,
        "binding":      example_binding_energy_single_frame,
        "trajectory":   example_trajectory_binding,
        "decomp":       example_decomposition,
        "mdcrd":        example_mdcrd_reader,
        "subtopo":      example_subtopology,
        "solvated":     example_solvated_binding,
        "qh":           example_quasi_harmonic,
    }

    if len(sys.argv) > 1 and sys.argv[1] in examples:
        examples[sys.argv[1]]()
    else:
        print("MM-GBSA Examples")
        print("================")
        print(f"Usage: python {sys.argv[0]} <example>\n")
        print("Available examples:")
        for name, fn in examples.items():
            print(f"  {name:12s} - {fn.__doc__.strip().splitlines()[0]}")
        print()
        # Run single-frame demo by default
        example_single_frame()
