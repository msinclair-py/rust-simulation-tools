---
name: rst-mmpbsa
description: Compute binding free energies with rust_simulation_tools' MM-PBSA / MM-GBSA implementation — trajectory-averaged ΔG with Generalized Born or Poisson-Boltzmann solvation, single-frame energies, per-residue decomposition, and entropy corrections. Use for "binding free energy", "MM-PBSA", "MM-GBSA", "ΔG of binding", "GB/PB solvation energy", "per-residue decomposition", or "interaction entropy".
---

# MM-PBSA / MM-GBSA binding free energy

Computes ΔG_bind = ΔE_MM + ΔG_solv(GB or PB) + ΔG_SA between a receptor and a
ligand defined by residue indices. Works on solvated **or** dry topologies —
solvent is automatically stripped based on the receptor/ligand selection, so
residue indices always refer to the **original (full) topology** numbering.

## Parameter objects

```python
from rust_simulation_tools import GbModel, GbParams, PbParams, SaParams

gb = GbParams(model=GbModel.ObcII, salt_concentration=0.15)   # GB (fast)
pb = PbParams(grid_spacing=0.5, solvent_dielectric=80.0, salt_concentration=0.15)  # PB
sa = SaParams()   # nonpolar surface-area term (defaults are reasonable)
```
Use `gb_params=` for MM-GBSA or `pb_params=` for MM-PBSA — pass one, not both.

## Trajectory-averaged binding energy

```python
topo = rst.read_prmtop("system.prmtop")
receptor = list(range(0, 250))
ligand   = list(range(250, 251))

result = rst.compute_binding_energy(
    topo,
    trajectory_path="trajectory.dcd",
    receptor_residues=receptor,
    ligand_residues=ligand,
    gb_params=gb, sa_params=sa,        # or pb_params=pb
    trajectory_format="dcd",           # "dcd" or "mdcrd"
)

result.mean_delta_total, result.std_delta_total   # kcal/mol
result.mean_delta_mm, result.mean_delta_gb, result.mean_delta_sa
result.frames                                      # per-frame FrameEnergy list
```
Single frame: `rst.compute_binding_energy_single_frame(topo, coords, receptor_residues, ligand_residues, ...)`.

Count atoms actually used (handy to confirm solvent exclusion):
```python
sel = topo.build_selection(receptor)     # -> dict with "atom_indices"
n = len(sel["atom_indices"])
```

## Per-residue decomposition

```python
decomp = rst.decompose_binding_energy(
    topo, coords,
    receptor_residues=receptor, ligand_residues=ligand,
)
for res in sorted(decomp.receptor_residues, key=lambda r: r.total())[:5]:
    print(f"{res.residue_label}{res.residue_index}: {res.total():.2f} kcal/mol")
# also decomp.ligand_residues; each ResidueContribution has .total() and components
```

## Individual energy terms (lower level)

```python
rst.compute_mm_energy(topo, coords)            # -> MmEnergy
rst.compute_gb_energy(topo, coords, gb)        # -> GbEnergy
rst.compute_pb_energy(topo, coords, pb)        # -> PbEnergy
rst.compute_sa_energy(topo, coords, sa)        # -> SaEnergy
```

## Entropy corrections

```python
rst.interaction_entropy(frames, temperature=298.15)   # -TΔS via interaction-entropy method
rst.quasi_harmonic_entropy(...)                        # quasi-harmonic estimate
```

## Scaling out

`compute_binding_energy` releases the GIL during the heavy work, so it
parallelizes well across systems with Parsl/multiprocessing — see
`examples/example_parsl_mmpbsa.py`. Core example: `examples/example_mmpbsa.py`.
