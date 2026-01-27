"""Interaction fingerprint example - compute per-residue LJ and electrostatic energies.

This example demonstrates two approaches:
1. The new simplified FingerprintSession API (recommended)
2. The original low-level compute_fingerprints API (for backward compatibility)
"""

import numpy as np


def example_fingerprint_session():
    """New simplified API using FingerprintSession.

    Reduces Python boilerplate from 25+ lines to ~5 lines while supporting
    both target and binder fingerprinting modes.
    """
    from rust_simulation_tools import FingerprintSession, FingerprintMode

    # Create session - topology params extracted automatically
    session = FingerprintSession("system.prmtop", "trajectory.dcd")

    # Define selections
    n_target = 10
    session.set_target_residues(range(n_target))  # Residues 0-9
    session.set_binder_residues(range(n_target, session.n_residues))  # Rest

    print(f"Computing fingerprints for {n_target} target residues")
    print(f"Interacting with {session.n_residues - n_target} binder residues")
    print(f"Processing {session.n_frames} frames\n")

    # -------------------------------------------------------------------------
    # Fingerprint target residues (default mode)
    # -------------------------------------------------------------------------
    print("=== Target Fingerprints ===")
    target_lj_fps = []
    target_es_fps = []

    for lj_fp, es_fp in session:
        target_lj_fps.append(lj_fp)
        target_es_fps.append(es_fp)

    target_lj_fps = np.array(target_lj_fps)
    target_es_fps = np.array(target_es_fps)

    print(f"Per-residue mean LJ energies (kJ/mol):")
    for i, label in enumerate(session.residue_labels):
        print(f"  {label}: {np.mean(target_lj_fps[:, i]):.2f}")

    print(f"\nPer-residue mean ES energies (kJ/mol):")
    for i, label in enumerate(session.residue_labels):
        print(f"  {label}: {np.mean(target_es_fps[:, i]):.2f}")

    print(f"\nTotal target interaction energy: "
          f"LJ={np.sum(target_lj_fps[-1]):.2f}, "
          f"ES={np.sum(target_es_fps[-1]):.2f} kJ/mol")

    # -------------------------------------------------------------------------
    # Fingerprint binder residues (switch mode)
    # -------------------------------------------------------------------------
    print("\n=== Binder Fingerprints ===")
    session.set_fingerprint_mode(FingerprintMode.Binder)
    session.seek(0)  # Reset to first frame

    binder_lj_fps = []
    binder_es_fps = []

    for lj_fp, es_fp in session:
        binder_lj_fps.append(lj_fp)
        binder_es_fps.append(es_fp)

    binder_lj_fps = np.array(binder_lj_fps)
    binder_es_fps = np.array(binder_es_fps)

    print(f"Binder residue count: {len(session.residue_labels)}")
    print(f"First 5 binder residues:")
    for i, label in enumerate(session.residue_labels[:5]):
        print(f"  {label}: LJ={np.mean(binder_lj_fps[:, i]):.2f}, "
              f"ES={np.mean(binder_es_fps[:, i]):.2f}")

    print(f"\nTotal binder interaction energy: "
          f"LJ={np.sum(binder_lj_fps[-1]):.2f}, "
          f"ES={np.sum(binder_es_fps[-1]):.2f} kJ/mol")

    # Energy symmetry: sum(target) should equal sum(binder)
    print("\n=== Energy Symmetry Check ===")
    target_total = np.sum(target_lj_fps[-1]) + np.sum(target_es_fps[-1])
    binder_total = np.sum(binder_lj_fps[-1]) + np.sum(binder_es_fps[-1])
    print(f"Target total: {target_total:.2f} kJ/mol")
    print(f"Binder total: {binder_total:.2f} kJ/mol")
    print(f"Difference: {abs(target_total - binder_total):.6f} kJ/mol")


def example_low_level_api():
    """Original low-level API using compute_fingerprints directly.

    This approach provides more control but requires more boilerplate.
    """
    from rust_simulation_tools import compute_fingerprints, read_prmtop, DcdReader

    # Load AMBER topology for force field parameters
    topo = read_prmtop("system.prmtop")

    # Get charges and LJ parameters
    charges = np.array(topo.charges())
    sigmas = np.array(topo.sigmas())
    epsilons = np.array(topo.epsilons())

    # Build residue map (atom indices grouped by residue)
    resmap_indices, resmap_offsets = topo.build_resmap()
    resmap_indices = np.array(resmap_indices, dtype=np.int64)
    resmap_offsets = np.array(resmap_offsets, dtype=np.int64)

    # Define target residues (e.g., first 10 residues of protein)
    n_target = 10
    target_resmap_indices = resmap_indices[:resmap_offsets[n_target]]
    target_resmap_offsets = resmap_offsets[:n_target + 1]

    # Define binder atoms (all atoms after target residues)
    binder_indices = np.arange(resmap_offsets[n_target], topo.n_atoms, dtype=np.int64)

    # Load trajectory
    reader = DcdReader("trajectory.dcd")
    print(f"Computing fingerprints for {n_target} residues over {reader.n_frames} frames")

    # Compute fingerprints for each frame
    lj_fingerprints = []
    es_fingerprints = []

    while True:
        frame = reader.read_frame()
        if frame is None:
            break
        positions, _ = frame

        lj_fp, es_fp = compute_fingerprints(
            positions, charges, sigmas, epsilons,
            target_resmap_indices, target_resmap_offsets, binder_indices
        )
        lj_fingerprints.append(lj_fp)
        es_fingerprints.append(es_fp)

    lj_fingerprints = np.array(lj_fingerprints)
    es_fingerprints = np.array(es_fingerprints)

    # Summarize results
    print(f"\nPer-residue mean LJ energies (kJ/mol):")
    for i in range(n_target):
        resname = topo.residue_labels[i]
        print(f"  {resname}: {np.mean(lj_fingerprints[:, i]):.2f}")

    print(f"\nPer-residue mean ES energies (kJ/mol):")
    for i in range(n_target):
        resname = topo.residue_labels[i]
        print(f"  {resname}: {np.mean(es_fingerprints[:, i]):.2f}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--low-level":
        print("Running low-level API example\n")
        example_low_level_api()
    else:
        print("Running FingerprintSession example (recommended)\n")
        print("Use --low-level flag for original API\n")
        example_fingerprint_session()
