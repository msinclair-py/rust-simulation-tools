# Rust Simulation Tools

[![CI/CD](https://github.com/msinclair-py/rust-simulation-tools/workflows/CI%2FCD/badge.svg)](https://github.com/msinclair-py/rust-simulation-tools/actions)
[![PyPI version](https://badge.fury.io/py/rust-simulation-tools.svg)](https://badge.fury.io/py/rust-simulation-tools)
[![Python versions](https://img.shields.io/pypi/pyversions/rust-simulation-tools.svg)](https://pypi.org/project/rust-simulation-tools/)

Fast, numerically stable MD trajectory processing implemented in Rust with a clean python API. 
Installable via `pip`, integrates smoothly with MDAnalysis or mdtraj, and ships with tests.

## Installation

```bash
pip install rust-simulation-tools
```

## Features

- ⚡ **Fast**: Rust implementation with SIMD optimizations
- 🎯 **Accurate**: Numerically stable Kabsch alignment and fragment-based unwrapping
- 🧪 **Well tested**: Comprehensive test suite with >80% coverage
- 🧩 **Easy integration**: Works directly with MDAnalysis selections/indices

## Usage

```python
import MDAnalysis as mda
from rust_simulation_tools import kabsch_align

# Load trajectory
u = mda.Universe("topology.pdb", "trajectory.dcd")

# Select alignment atoms
align_selection = u.select_atoms("backbone")
align_indices = align_selection.indices

# Get coordinates
reference = u.atoms.positions.copy()
trajectory = np.array([ts.positions for ts in u.trajectory])

# Align
aligned = kabsch_align(trajectory, reference, align_indices)
```

## API Reference

```python
kabsch_align(
    trajectory: np.ndarray,      # float, shape [n_frames, n_atoms, 3]
    reference: np.ndarray,       # float, shape [n_atoms, 3]
    align_idx: np.ndarray        # int,   shape [n_alignment_atoms]
) -> np.ndarray                  # float, shape [n_frames, n_atoms, 3]

unwrap_system(
    trajectory: np.ndarray,      # float, shape [n_frames, n_atoms, 3]
    box_dimensions: np.ndarray,  # float, shape [n_frames, 3]
    fragment_idx: np.ndarray     # int,   shape [n_atoms]
) -> np.ndarray                  # float, shape [n_frames, n_atoms, 3]
```

## Development

```bash
# Clone repository
git clone https://github.com/msinclair-py/rust-simulation-tools.git
cd rust-simulation-tools

# Install development dependencies
pip install maturin pytest pytest-cov numpy

# Build and install in development mode
maturin develop --release

# Install with pip
pip install target/wheels/name-of-package.whl

# Run tests
pytest tests/ -v --cov
```

## License

MIT License
