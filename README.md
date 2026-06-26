# Rust Simulation Tools

[![CI/CD](https://github.com/msinclair-py/rust-simulation-tools/workflows/CI%2FCD/badge.svg)](https://github.com/msinclair-py/rust-simulation-tools/actions)
[![PyPI version](https://img.shields.io/pypi/v/rust-simulation-tools)](https://pypi.org/project/rust-simulation-tools/)

High-performance molecular dynamics analysis library with a Python API. Written in Rust for speed, exposed to Python via PyO3.

## Installation
From PyPI using uv:
```bash
uv pip install rust-simulation-tools
```

For the latest version which may not be on PyPI yet, make sure you have cloned this repo and have
`maturin` installed to your environment of choice:
```bash
uv venv /path/to/env
source /path/to/env/bin/activate
uv pip install maturin

git clone https://github.com/msinclair-py/rust-simulation-tools.git
cd rust-simulation-tools
maturin develop --release
```


## Features

- **File I/O**: AMBER topology/coordinates, DCD and MDCRD trajectories; PDB, mmCIF, mol2, SDF structures
- **System building**: tleap-style `SystemBuilder` — force fields, solvation, ions, prmtop/inpcrd output
- **Ligand parameterization**: built-in antechamber — GAFF2 atom typing + AM1-BCC charges (no AmberTools needed)
- **Selections**: VMD-style atom selection with property access
- **Analysis**: SASA, trajectory unwrapping, Kabsch alignment
- **Fingerprints**: Per-residue interaction energies (LJ + electrostatic)
- **Minimization**: Steepest-descent + conjugate-gradient with optional restraints
- **MM-PBSA/GBSA**: Binding free energy with per-residue decomposition
- **Interface scoring**: ipSAE, pDockQ, pDockQ2, LIS, ipTM for predicted complexes

## Quick Start

### Load a System

```python
from rust_simulation_tools import read_prmtop, read_inpcrd, DcdReader

# Load topology and coordinates
topo = read_prmtop("system.prmtop")
coords, box = read_inpcrd("system.inpcrd")

# Load trajectory
dcd = DcdReader("trajectory.dcd")
trajectory, boxes = dcd.read_all()
```

### Build a System

The `SystemBuilder` parameterizes structures and writes simulation-ready AMBER
files — no AmberTools install required.

```python
from rust_simulation_tools import SystemBuilder

builder = SystemBuilder()
builder.load_protein_ff19sb()   # protein force field
builder.load_gaff2()            # small-molecule force field
builder.load_water_opc()        # OPC water model

# Load structures (PDB / mmCIF for proteins, mol2/SDF for ligands)
protein = builder.load_pdb("protein.pdb")
ligand  = builder.load_ligand("ligand.sdf", net_charge=0)  # auto GAFF2 + AM1-BCC

# Combine, solvate, ionize
system = builder.combine([protein, ligand])
builder.solvate_box(system, buffer=12.0)
builder.add_salt(system, "Na+", "Cl-", concentration=0.150)  # neutralize + 150 mM

# Write output
builder.write_amber(system, "complex.prmtop", "complex.inpcrd")
builder.write_pdb(system, "complex.pdb")
```

For implicit solvent, skip `load_water_opc`/solvation and write the topology
directly. See `examples/example_explicit_solvent.py`,
`example_implicit_solvent.py`, and `example_protein_ligand.py`.

### Parameterize a Ligand

```python
import rust_simulation_tools as rst

# Standalone: write a parameterized mol2 (GAFF2 types + charges)
rst.parameterize_ligand(
    "ligand.sdf", "ligand_gaff2.mol2",
    net_charge=0,
    charge_method="am1bcc",   # or "gasteiger" (faster, less accurate)
)

# Raw AM1 Mulliken charges from atomic numbers + coordinates
import numpy as np
charges = rst.compute_am1_charges(
    np.array([8, 1, 1], dtype=np.int64),          # O, H, H
    np.array([[0, 0, 0], [0, 0.757, 0.587], [0, -0.757, 0.587]], dtype=float),
    charge=0,
)
```

### Atom Selection

Select atoms using VMD-style expressions. The `select()` method returns a `Selection` object with direct property access.

```python
# Select protein backbone
backbone = topo.select("backbone")
print(f"{backbone.n_atoms} atoms, {backbone.n_residues} residues")

# Access properties directly
ca = topo.select("protein and name CA")
print(ca.masses)          # numpy array of masses
print(ca.charges)         # numpy array of charges
print(ca.total_mass())    # sum of masses

# Distance-based selection (requires coordinates)
near_lig = topo.select("protein and within 5.0 of resname LIG", coordinates=coords)

# Set operations
charged = topo.select("charge < -0.5 or charge > 0.5")
sidechain = topo.select("sidechain")
charged_sidechain = sidechain & charged  # intersection
```

**Supported selection keywords:**
- `protein`, `backbone`, `sidechain`, `water`
- `name CA`, `resname ALA`, `resid 1-50`
- `charge > 0.5`, `mass < 2.0`
- `within 5.0 of resname LIG`
- Boolean: `and`, `or`, `not`

### SASA Calculation

```python
from rust_simulation_tools import compute_sasa_from_topology

# Single frame
sasa = compute_sasa_from_topology(topo, coords)
print(f"Total SASA: {sasa['total']:.1f} A^2")          # float
print(f"Per-atom: {sasa['per_atom'].shape}")           # ndarray (n_atoms,)
print(f"Per-residue: {sasa['per_residue'].shape}")     # ndarray (n_residues,), by residue index

# Trajectory: total -> ndarray (n_frames,), per_residue -> list of per-frame dicts
traj_sasa = compute_sasa_trajectory_from_topology(topo, trajectory)
```

### Trajectory Alignment

```python
from rust_simulation_tools import kabsch_align

# Align trajectory to first frame using backbone atoms
backbone = topo.select("backbone")
aligned = kabsch_align(trajectory, trajectory[0], backbone.indices)
```

### Trajectory Unwrapping

```python
from rust_simulation_tools import unwrap_dcd

# Remove periodic boundary artifacts
unwrapped, boxes = unwrap_dcd("trajectory.dcd")
```

### Interaction Fingerprints

Calculate per-residue LJ and electrostatic interactions between a target and partner.

```python
from rust_simulation_tools import FingerprintSession, FingerprintMode

session = FingerprintSession("system.prmtop", "trajectory.dcd")
session.set_target_residues(range(10))           # residues to fingerprint
session.set_binder_residues(range(10, 100))      # interaction partner

# Iterate over frames
for lj_fp, es_fp in session:
    print(f"LJ: {lj_fp.sum():.2f}, ES: {es_fp.sum():.2f} kJ/mol")

# Switch perspective: fingerprint binder residues instead
session.set_fingerprint_mode(FingerprintMode.Binder)
session.seek(0)
```

### MM-PBSA/GBSA Binding Energy

Calculate binding free energy with Generalized Born or Poisson-Boltzmann solvation.

```python
from rust_simulation_tools import (
    compute_binding_energy,
    decompose_binding_energy,
    GbModel, GbParams, PbParams, SaParams,
)

# MM-GBSA over trajectory
result = compute_binding_energy(
    topo,
    trajectory_path="trajectory.dcd",
    receptor_residues=list(range(0, 250)),
    ligand_residues=list(range(250, 251)),
    gb_params=GbParams(model=GbModel.ObcII, salt_concentration=0.15),
    sa_params=SaParams(),
    trajectory_format="dcd",
)

print(f"Delta G: {result.mean_delta_total:.2f} +/- {result.std_delta_total:.2f} kcal/mol")
print(f"  MM:  {result.mean_delta_mm:.2f}")
print(f"  GB:  {result.mean_delta_gb:.2f}")
print(f"  SA:  {result.mean_delta_sa:.2f}")

# MM-PBSA (use pb_params instead of gb_params)
pb_result = compute_binding_energy(
    topo, "trajectory.dcd",
    receptor_residues=list(range(0, 250)),
    ligand_residues=list(range(250, 251)),
    pb_params=PbParams(grid_spacing=0.5, salt_concentration=0.15),
)

# Per-residue decomposition
decomp = decompose_binding_energy(
    topo, coords,
    receptor_residues=list(range(0, 250)),
    ligand_residues=list(range(250, 251)),
)

for res in sorted(decomp.receptor_residues, key=lambda r: r.total())[:5]:
    print(f"{res.residue_label}{res.residue_index}: {res.total():.2f} kcal/mol")
```

### Energy Minimization

Steepest-descent + conjugate-gradient minimization with optional positional
restraints and a full energy-component breakdown.

```python
from rust_simulation_tools import minimize, MinimizeConfig

config = MinimizeConfig(
    max_cycles=5000,
    sd_cycles=100,           # initial steepest-descent steps
    convergence_rms=0.01,
    cutoff=10.0,
    restraint_mask="backbone",  # optional; omit for unrestrained
    restraint_weight=10.0,
)

result = minimize("system.prmtop", "system.inpcrd", config=config, output="min.inpcrd")
print(f"Energy: {result.final_energy:.2f} kcal/mol  converged={result.converged}")

ec = result.energy_components       # bond, angle, dihedral, vdw,
print(ec.total(), ec.vdw, ec.elec_recip)   # elec_direct, elec_recip, vdw_14, elec_14
```

Use `minimize_topology(topo, "system.inpcrd", ...)` to reuse a pre-loaded
topology. See `examples/example_minimization.py`.

### Interface Scoring (ipSAE)

Score predicted complexes (AlphaFold-Multimer, Boltz, Chai) from pLDDT and PAE.

```python
import numpy as np
from rust_simulation_tools import compute_ipsae

plddt = np.load("plddt.npy")            # per-residue, 0-100 scale, shape (N,)
pae = np.load("pae.npy").flatten()      # predicted aligned error, flattened (N*N,)

results = compute_ipsae("model.pdb", plddt, pae)   # PDB or CIF
for pair in results["max_pairs"]:       # also "directed_pairs"
    print(f"{pair['chain1']}-{pair['chain2']}: "
          f"ipSAE={pair['ipSAE']:.3f} pDockQ={pair['pDockQ']:.3f} LIS={pair['LIS']:.3f}")
```

`compute_ipsae_from_arrays(coords, chains, chain_types, plddt, pae)` does the same
from in-memory arrays. See `examples/example_ipsae.py`.

## API Reference

### File I/O

| Function | Description |
|----------|-------------|
| `read_prmtop(path)` | Load AMBER topology, returns `AmberTopology` |
| `read_inpcrd(path)` | Load AMBER coordinates, returns `(coords, box)` |
| `DcdReader(path)` | DCD trajectory reader |
| `MdcrdReader(path, n_atoms, has_box)` | AMBER ASCII trajectory reader |

### System Building

| Method | Description |
|--------|-------------|
| `SystemBuilder()` | Create a tleap-style builder |
| `.load_protein_ff19sb()`, `.load_gaff2()`, `.load_water_opc()` | Load force fields / water model |
| `.load_custom_frcmod(path)`, `.load_custom_lib(path)` | Load custom ligand parameters |
| `.load_pdb(path)`, `.load_mmcif(path)`, `.load_mol2(path)` | Load a structure, returns `System` |
| `.load_ligand(path, net_charge=0)` | Load + parameterize a ligand (GAFF2 + AM1-BCC) |
| `.combine([systems])` | Merge systems into one `System` |
| `.solvate_box(system, buffer=12.0, closeness=1.0)` | Solvate in an OPC water box |
| `.add_ions(system, ion, count=None)` | Add ions (`"neutralize"`, int count, or float conc.) |
| `.add_salt(system, cation="Na+", anion="Cl-", concentration=0.150)` | Neutralize + add salt |
| `.write_amber(system, prmtop, inpcrd)`, `.write_prmtop/.write_inpcrd/.write_pdb` | Write output |

`System` exposes `.n_atoms`, `.n_residues`, `.total_charge`, `.box_dimensions`,
`.box_angles`.

### Parameterization

| Function | Description |
|----------|-------------|
| `parameterize_ligand(input, output, net_charge=0, charge_method="am1bcc")` | Write a GAFF2/charge-assigned mol2 (`"am1bcc"` or `"gasteiger"`) |
| `compute_am1_charges(atomic_numbers, coords, charge=0)` | Raw AM1 Mulliken charges |

### AmberTopology

| Property/Method | Description |
|-----------------|-------------|
| `.n_atoms`, `.n_residues` | System size |
| `.atom_names`, `.residue_labels` | Atom/residue names |
| `.charges()`, `.sigmas()`, `.epsilons()` | Force field parameters |
| `.select(expression, coordinates=None)` | VMD-style selection, returns `Selection` |
| `.bonds()` | List of bonded atom pairs |

### Selection

| Property/Method | Description |
|-----------------|-------------|
| `.n_atoms`, `.n_residues` | Selection size |
| `.indices` | Atom indices (numpy array) |
| `.masses`, `.charges`, `.radii` | Per-atom properties |
| `.atom_names`, `.residue_names` | Names as lists |
| `.positions` | Coordinates (if provided during selection) |
| `.total_mass()`, `.total_charge()` | Aggregate properties |
| `&`, `\|`, `-` | Set operations (intersection, union, difference) |

### DcdReader

| Property/Method | Description |
|-----------------|-------------|
| `.n_frames`, `.n_atoms` | Trajectory size |
| `.read_frame()` | Read next frame, returns `(coords, box)` |
| `.read_all()` | Read all frames, returns `(trajectory, boxes)` |
| `.seek(frame)` | Jump to frame index |

### Analysis Functions

| Function | Description |
|----------|-------------|
| `compute_sasa_from_topology(topo, coords)` | SASA using topology for radii |
| `compute_sasa_trajectory_from_topology(topo, trajectory)` | Per-frame SASA over a trajectory |
| `calculate_sasa(coords, radii, residue_indices)` | SASA with explicit radii |
| `kabsch_align(trajectory, reference, align_indices)` | RMSD-minimizing alignment |
| `unwrap_dcd(path)` | Remove PBC artifacts from DCD |
| `unwrap_system(trajectory, boxes)` | Remove PBC artifacts |
| `FingerprintSession(prmtop, trajectory)` | Per-residue LJ/electrostatic fingerprints |

### Minimization

| Function | Description |
|----------|-------------|
| `minimize(prmtop, inpcrd, config=None, output=None)` | Minimize from files, returns `MinimizeResult` |
| `minimize_topology(topo, inpcrd, config=None, output=None)` | Minimize with a pre-loaded topology |
| `MinimizeConfig(max_cycles, sd_cycles, convergence_rms, cutoff, restraint_mask, restraint_weight, initial_step_size)` | Minimization settings |

`MinimizeResult` exposes `.final_energy`, `.final_rms`, `.cycles`, `.converged`,
`.energy_components` (`.bond`, `.angle`, `.dihedral`, `.vdw`, `.elec_direct`,
`.elec_recip`, `.vdw_14`, `.elec_14`, `.total()`).

### MM-PBSA/GBSA

| Function | Description |
|----------|-------------|
| `compute_binding_energy(...)` | Trajectory-averaged binding energy |
| `compute_binding_energy_single_frame(...)` | Single frame binding energy |
| `decompose_binding_energy(...)` | Per-residue energy decomposition |
| `compute_mm_energy(topo, coords)` | Molecular mechanics energy |
| `compute_gb_energy(topo, coords, params)` | GB solvation energy |
| `compute_pb_energy(topo, coords, params)` | PB solvation energy |
| `compute_sa_energy(topo, coords, params)` | Nonpolar surface-area energy |
| `interaction_entropy(frames, temperature)` | Interaction-entropy correction |
| `quasi_harmonic_entropy(...)` | Quasi-harmonic entropy estimate |

Parameter objects: `GbParams`, `PbParams`, `SaParams`, `GbModel`.

### Interface Scoring

| Function | Description |
|----------|-------------|
| `compute_ipsae(structure_path, plddt, pae, pdockq_cutoff=8.0, pae_cutoff=12.0)` | ipSAE/pDockQ/LIS/ipTM from a PDB/CIF file |
| `compute_ipsae_from_arrays(coords, chains, chain_types, plddt, pae, ...)` | Same, from in-memory arrays |

## Agent Skills

The [`skills/`](skills) directory contains [agent skills](skills/README.md) that
teach AI coding assistants (e.g. Claude Code) how to use this package — system
building, ligand parameterization, trajectory analysis, MM-PBSA, minimization,
and ipSAE scoring. They are mirrored under `.claude/skills/` so they load
automatically when working in this repo. See [`skills/README.md`](skills/README.md)
for the full list and how to install them elsewhere.

## Development

```bash
git clone https://github.com/msinclair-py/rust-simulation-tools.git
cd rust-simulation-tools
pip install maturin pytest numpy
maturin develop --release
pytest tests/ -v
```

## License

MIT License
