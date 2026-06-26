---
name: rst-ipsae
description: Score predicted protein-protein interfaces with rust_simulation_tools — compute ipSAE, pDockQ, pDockQ2, LIS, and ipTM from a PDB/CIF structure plus pLDDT and PAE, or directly from in-memory arrays. Use for "ipSAE", "pDockQ", "interface confidence", "scoring AlphaFold/Boltz/Chai complexes", "PAE-based interface score", or "LIS".
---

# ipSAE interface scoring

Computes interface-quality metrics for predicted complexes (AlphaFold-Multimer,
Boltz, Chai, etc.) from per-residue confidence (pLDDT) and the predicted aligned
error (PAE) matrix. Metrics: pDockQ, pDockQ2, LIS, ipTM, and ipSAE.

## From a structure file

```python
import numpy as np
from rust_simulation_tools import compute_ipsae

# pLDDT: per-residue confidence on a 0-100 scale, shape (N,)
plddt = np.load("confidences.npz")["plddt"] * 100.0   # multiply if yours is 0-1 (e.g. Boltz)

# PAE: predicted aligned error (N, N), flattened row-major to 1-D
pae = np.load("pae.npz")["pae"].flatten()             # shape (N*N,)

results = compute_ipsae("model.pdb", plddt, pae)      # PDB or CIF
```

`results` is a dict with two keys:
- `"directed_pairs"` — every ordered chain pair (A→B and B→A separately)
- `"max_pairs"` — one entry per unordered pair, keeping the higher-ipSAE direction

```python
for p in results["directed_pairs"]:
    print(p["chain1"], "->", p["chain2"],
          p["pDockQ"], p["pDockQ2"], p["LIS"], p["ipTM"], p["ipSAE"])
```

## From in-memory arrays (no file I/O)

Use when coordinates/chains already come from MDAnalysis, BioPython, etc.:

```python
from rust_simulation_tools import compute_ipsae_from_arrays

coords = ...                       # (N, 3) CA / C1' coordinates, Angstroms
chains = ["A", "A", "A", "B", "B"] # per-residue chain label, length N
chain_types = {"A": "protein", "B": "protein"}   # or "nucleic_acid"
results = compute_ipsae_from_arrays(
    coords, chains, chain_types, plddt, pae,
    pdockq_cutoff=8.0,   # contact distance cutoff (default 8 A)
    pae_cutoff=12.0,     # PAE cutoff for LIS/ipSAE (default 12 A)
)
```
`compute_ipsae` takes the same optional `pdockq_cutoff` / `pae_cutoff` arguments.

## Key points

- pLDDT must be on the **0-100** scale — convert 0-1 confidences first.
- PAE must be **flattened** to 1-D, row-major, length `N*N`.
- All per-residue arrays share the same length/ordering as the structure.

Full example: `examples/example_ipsae.py`.
