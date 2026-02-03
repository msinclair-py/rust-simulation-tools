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

- **File I/O**: AMBER topology/coordinates, DCD and MDCRD trajectories
- **Selections**: VMD-style atom selection with property access
- **Analysis**: SASA, trajectory unwrapping, Kabsch alignment
- **Fingerprints**: Per-residue interaction energies (LJ + electrostatic)
- **MM-PBSA/GBSA**: Binding free energy with per-residue decomposition

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
print(f"Total SASA: {sasa['total']:.1f} A^2")
print(f"Per-atom: {sasa['per_atom'].shape}")
print(f"Per-residue: {sasa['per_residue']}")  # dict of residue_idx -> area
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

## API Reference

### File I/O

| Function | Description |
|----------|-------------|
| `read_prmtop(path)` | Load AMBER topology, returns `AmberTopology` |
| `read_inpcrd(path)` | Load AMBER coordinates, returns `(coords, box)` |
| `DcdReader(path)` | DCD trajectory reader |
| `MdcrdReader(path, n_atoms, has_box)` | AMBER ASCII trajectory reader |

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
| `calculate_sasa(coords, radii, residue_indices)` | SASA with explicit radii |
| `kabsch_align(trajectory, reference, align_indices)` | RMSD-minimizing alignment |
| `unwrap_dcd(path)` | Remove PBC artifacts from DCD |
| `unwrap_system(trajectory, boxes)` | Remove PBC artifacts |

### MM-PBSA/GBSA

| Function | Description |
|----------|-------------|
| `compute_binding_energy(...)` | Trajectory-averaged binding energy |
| `compute_binding_energy_single_frame(...)` | Single frame binding energy |
| `decompose_binding_energy(...)` | Per-residue energy decomposition |
| `compute_mm_energy(topo, coords)` | Molecular mechanics energy |
| `compute_gb_energy(topo, coords, params)` | GB solvation energy |
| `compute_pb_energy(topo, coords, params)` | PB solvation energy |
| `interaction_entropy(frames, temperature)` | Entropy correction |

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
