# Rust Simulation Tools

[![CI/CD](https://github.com/msinclair-py/rust-simulation-tools/workflows/CI%2FCD/badge.svg)](https://github.com/msinclair-py/rust-simulation-tools/actions)
[![PyPI version](https://img.shields.io/pypi/v/rust-simulation-tools)](https://pypi.org/project/rust-simulation-tools/)

Fast MD trajectory processing and analysis in Rust with a Python API.

## Installation

```bash
pip install rust-simulation-tools
```

## Features

- Kabsch alignment with SIMD optimizations
- Fragment-based periodic boundary unwrapping
- SASA calculation (Shrake-Rupley with KD-tree acceleration)
- Interaction energy fingerprinting (LJ + electrostatic)
- AMBER file readers (prmtop, inpcrd)
- DCD trajectory reader with streaming support

## Quick Examples

### Trajectory Alignment

```python
from rust_simulation_tools import kabsch_align

aligned = kabsch_align(trajectory, reference, align_indices)
```

### SASA Calculation

```python
from rust_simulation_tools import calculate_sasa, get_radii_array

radii = get_radii_array(elements)  # ['C', 'N', 'O', ...]
result = calculate_sasa(coords, radii, residue_indices)
# result['total'], result['per_atom'], result['per_residue']
```

### AMBER Topology + DCD Trajectory

```python
from rust_simulation_tools import read_prmtop, DcdReader

topo = read_prmtop("system.prmtop")
charges, sigmas, epsilons = topo.charges(), topo.sigmas(), topo.epsilons()

dcd = DcdReader("trajectory.dcd")
for i in range(dcd.n_frames):
    coords, box = dcd.read_frame()
```

### Interaction Fingerprints

```python
from rust_simulation_tools import compute_fingerprints, read_prmtop

topo = read_prmtop("system.prmtop")
resmap_indices, resmap_offsets = topo.build_resmap()

lj_fp, es_fp = compute_fingerprints(
    positions, topo.charges(), topo.sigmas(), topo.epsilons(),
    resmap_indices, resmap_offsets, binder_indices
)
```

## API Reference

```python
# Alignment & unwrapping
kabsch_align(trajectory, reference, align_idx) -> aligned_trajectory
unwrap_system(trajectory, box_dimensions, fragment_idx) -> unwrapped_trajectory

# SASA
calculate_sasa(coords, radii, residue_indices, probe_radius=1.4) -> dict
calculate_sasa_trajectory(trajectory, radii, residue_indices) -> dict
calculate_total_sasa(coords, radii, probe_radius=1.4) -> float
get_vdw_radius(element) -> float
get_radii_array(elements) -> np.ndarray

# Fingerprinting
compute_fingerprints(positions, charges, sigmas, epsilons,
                     resmap_indices, resmap_offsets, binder_indices) -> (lj, es)

# File I/O
read_prmtop(path) -> AmberTopology
read_inpcrd(path) -> (positions, box_dimensions)
DcdReader(path)   # .n_frames, .n_atoms, .read_frame(), .seek(n), .read_all()
```

## Development

```bash
git clone https://github.com/msinclair-py/rust-simulation-tools.git
cd rust-simulation-tools
pip install maturin pytest pytest-cov numpy
maturin develop --release
pytest tests/ -v --cov
```

## License

MIT License
