# Rust Simulation Tools

[![CI/CD](https://github.com/msinclair-py/rust-simulation-tools/workflows/CI%2FCD/badge.svg)](https://github.com/msinclair-py/rust-simulation-tools/actions)
[![PyPI version](https://badge.fury.io/py/rust-simulation-tools.svg)](https://badge.fury.io/py/rust-simulation-tools)
[![Python versions](https://img.shields.io/pypi/pyversions/rust-simulation-tools.svg)](https://pypi.org/project/rust-simulation-tools/)

Fast Kabsch alignment for MD trajectories using Rust.

## Installation

```bash
pip install rust-simulation-tools
```

## Features

- ⚡ **Fast**: Rust implementation with SIMD optimizations
- 🔧 **Easy to use**: Simple Python API compatible with MDAnalysis
- 🧪 **Well tested**: Comprehensive test suite with >80% coverage
- 🎯 **Accurate**: Numerically stable Kabsch algorithm

## Usage

```python
import MDAnalysis as mda
from rust_simulation_tools import kabsch_align

# Load trajectory
u = mda.Universe("topology.pdb", "trajectory.dcd")

# Select alignment atoms
align_selection = u.select_atoms("backbone")
align_indices = align_selection.indices.astype(np.uintp)

# Get coordinates
reference = u.atoms.positions.copy().astype(np.float64)
trajectory = np.array([ts.positions for ts in u.trajectory], dtype=np.float64)

# Align
aligned = kabsch_align(trajectory, reference, align_indices)
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

# Run tests
pytest test_kabsch.py -v --cov
```

## License

MIT License
