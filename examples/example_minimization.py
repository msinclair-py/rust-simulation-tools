"""Energy minimization of a solvated system.

Demonstrates three minimization workflows:
1. Basic minimization with default settings.
2. Restrained minimization (backbone restraints) followed by unrestrained.
3. Minimization using a pre-loaded topology to avoid redundant file I/O.

Assumes you have already built a solvated system with prmtop/inpcrd files
(e.g. from the explicit solvent example).
"""

from rust_simulation_tools import (
    read_prmtop,
    minimize,
    minimize_topology,
    MinimizeConfig,
)

# =============================================================================
# Example 1: Basic minimization with default settings
# =============================================================================

print("=" * 60)
print("Example 1: Basic minimization (default config)")
print("=" * 60)

# minimize() loads the prmtop and inpcrd internally
result = minimize(
    "system.prmtop",
    "system.inpcrd",
    output="minimized.inpcrd",
)

print(f"  Final energy: {result.final_energy:.4f} kcal/mol")
print(f"  Final RMS:    {result.final_rms:.6f} kcal/(mol*A)")
print(f"  Cycles:       {result.cycles}")
print(f"  Converged:    {result.converged}")

# Access individual energy components
ec = result.energy_components
print(f"\n  Energy breakdown:")
print(f"    Bond:       {ec.bond:.4f}")
print(f"    Angle:      {ec.angle:.4f}")
print(f"    Dihedral:   {ec.dihedral:.4f}")
print(f"    VDW:        {ec.vdw:.4f}")
print(f"    Elec (dir): {ec.elec_direct:.4f}")
print(f"    Elec (PME): {ec.elec_recip:.4f}")
print(f"    1-4 VDW:    {ec.vdw_14:.4f}")
print(f"    1-4 Elec:   {ec.elec_14:.4f}")
print(f"    Total:      {ec.total():.4f}")

# =============================================================================
# Example 2: Two-stage minimization with restraints
# =============================================================================

print("\n" + "=" * 60)
print("Example 2: Restrained -> unrestrained minimization")
print("=" * 60)

# Stage 1: Minimize with backbone restraints (let solvent relax)
config_restrained = MinimizeConfig(
    max_cycles=2000,
    sd_cycles=500,
    convergence_rms=0.1,
    cutoff=10.0,
    restraint_mask="backbone",
    restraint_weight=10.0,
)

result1 = minimize(
    "system.prmtop",
    "system.inpcrd",
    config=config_restrained,
    output="minimized_stage1.inpcrd",
)
print(f"  Stage 1 (restrained):   E={result1.final_energy:.2f}, RMS={result1.final_rms:.4f}, cycles={result1.cycles}")

# Stage 2: Minimize without restraints (starting from stage 1 output)
config_free = MinimizeConfig(
    max_cycles=5000,
    sd_cycles=100,
    convergence_rms=0.01,
    cutoff=10.0,
)

result2 = minimize(
    "system.prmtop",
    "minimized_stage1.inpcrd",
    config=config_free,
    output="minimized_stage2.inpcrd",
)
print(f"  Stage 2 (unrestrained): E={result2.final_energy:.2f}, RMS={result2.final_rms:.4f}, cycles={result2.cycles}")

# =============================================================================
# Example 3: Using a pre-loaded topology
# =============================================================================

print("\n" + "=" * 60)
print("Example 3: Minimization with pre-loaded topology")
print("=" * 60)

# Load topology once and reuse for multiple operations
topo = read_prmtop("system.prmtop")
print(f"  Topology: {topo.n_atoms} atoms, {topo.n_residues} residues")

config = MinimizeConfig(
    max_cycles=1000,
    sd_cycles=200,
)

result = minimize_topology(
    topo,
    "system.inpcrd",
    config=config,
    output="minimized_final.inpcrd",
)

print(f"  Final energy: {result.final_energy:.4f} kcal/mol")
print(f"  Converged:    {result.converged}")
