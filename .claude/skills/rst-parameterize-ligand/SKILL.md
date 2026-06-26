---
name: rst-parameterize-ligand
description: Parameterize small-molecule ligands with rust_simulation_tools' built-in antechamber — assign GAFF2 atom types and AM1-BCC or Gasteiger partial charges, write a parameterized mol2, or compute raw AM1 Mulliken charges. Use for "parameterize a ligand", "GAFF2 atom types", "AM1-BCC charges", "antechamber", "Gasteiger charges", or "AM1 charges".
---

# Ligand parameterization (antechamber / AM1-BCC)

Pure-Rust reimplementation of antechamber: GAFF2 atom typing, BCC typing, and
AM1-BCC charges (AM1 semi-empirical Mulliken charges + bond-charge corrections),
matching AmberTools' `sqm` to <0.001 e. No external AmberTools install needed.

## Standalone: parameterize a ligand file

```python
import rust_simulation_tools as rst

rst.parameterize_ligand(
    input_path="ligand.sdf",        # SDF or mol2 input
    output_path="ligand_gaff2.mol2",# writes GAFF2 types + charges
    net_charge=0,                   # formal charge of the molecule
    charge_method="am1bcc",         # "am1bcc" (accurate) or "gasteiger" (fast)
)
```
Use `am1bcc` for production force fields; `gasteiger` only for quick/approximate
charges. The output mol2 carries GAFF2 atom types and per-atom charges ready for
`SystemBuilder.load_mol2`.

## Integrated with SystemBuilder

`builder.load_ligand("ligand.sdf", net_charge=0)` runs this parameterization
internally and returns a ready-to-combine `System`. See `rst-build-system`.

## Raw AM1 Mulliken charges

For advanced use (custom workflows, QM charge analysis), compute AM1 charges
directly from atomic numbers and coordinates:

```python
import numpy as np
atomic_numbers = np.array([8, 1, 1], dtype=np.int64)   # water: O, H, H
coords = np.array([
    [0.0,  0.000, 0.000],
    [0.0,  0.757, 0.587],
    [0.0, -0.757, 0.587],
])
charges = rst.compute_am1_charges(atomic_numbers, coords, charge=0)
# charges: ndarray of Mulliken charges, one per atom
```

Notes:
- `atomic_numbers` must be `int64`; `coords` are Angstroms, shape `(N, 3)`.
- `charge` is the total molecular charge.
- AM1 uses the NDDO approximation (overlap is identity for same-atom pairs), so
  Mulliken charge reduces to `q = Z - sum(P_diag)`.

For the equivalent AmberTools commands (`antechamber -c bcc -at gaff2`,
`parmchk2`), and how to feed a `.frcmod`/`.lib` into a build, see
`rst-build-system`. Full example: `examples/example_antechamber.py`.
