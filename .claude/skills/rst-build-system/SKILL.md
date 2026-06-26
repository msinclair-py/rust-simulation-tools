---
name: rst-build-system
description: Build simulation-ready molecular systems with rust_simulation_tools' SystemBuilder — load force fields, read PDB/mol2 structures, combine protein+ligand, solvate in explicit water, add counterions/salt, and write AMBER prmtop/inpcrd/PDB. Use for "build a system", "solvate", "add ions", "explicit/implicit solvent", "protein-ligand complex", or "write prmtop".
---

# Building systems with SystemBuilder

`SystemBuilder` is a tleap-style system builder. The pattern is: create the
builder, load force fields, load/combine structures into a `System`, optionally
solvate and ionize, then write output. Methods that add solvent/ions mutate the
`System` in place.

## Force fields (load before loading structures)

```python
builder = rst.SystemBuilder()
builder.load_protein_ff19sb()     # protein FF (ff19SB)
builder.load_gaff2()              # GAFF2 — needed for small-molecule ligands
builder.load_water_opc()          # OPC water model (for explicit solvent)
# Custom ligand parameters:
builder.load_custom_frcmod("ligand.frcmod")   # from parmchk2
builder.load_custom_lib("ligand.lib")         # custom residue library
```

## Loading structures → `System`

```python
protein = builder.load_pdb("protein.pdb")
ligand  = builder.load_mol2("ligand.mol2")     # pre-parameterized (GAFF2 + charges)
# Parameterize a raw ligand on the fly (runs antechamber internally):
ligand  = builder.load_ligand("ligand.sdf", net_charge=0)
system  = builder.combine([protein, ligand])
```

`System` exposes `.n_atoms`, `.n_residues`, `.total_charge`, `.box_dimensions`
(`None` until solvated).

## Solvation and ions

```python
builder.solvate_box(system, buffer=12.0, closeness=1.0)   # rectangular OPC box

# count="neutralize" balances net charge; an int adds that many ions;
# a float is interpreted as a molar concentration.
builder.add_ions(system, "Na+", count="neutralize")  # returns number added
builder.add_ions(system, "Na+", count=20)            # 20 explicit Na+
builder.add_ions(system, "Cl-", count=0.150)         # 150 mM Cl-
```

## Writing output

```python
builder.write_amber(system, "system.prmtop", "system.inpcrd")  # both at once
builder.write_prmtop(system, "system.prmtop")                  # or individually
builder.write_inpcrd(system, "system.inpcrd")
builder.write_pdb(system, "system.pdb")
```

## Workflows

- **Implicit solvent (GB):** load only `load_protein_ff19sb()`, `load_pdb`,
  then write prmtop/inpcrd. No solvation/ions — `box_dimensions` stays `None`.
- **Explicit solvent:** add `load_water_opc()`, `solvate_box`, then `add_ions`
  (neutralize, then excess salt), then `write_amber`.
- **Protein-ligand complex:** load protein + water + `load_gaff2()`, load custom
  `frcmod`/`lib` if needed, `load_mol2`/`load_ligand`, `combine`, solvate, ionize.

See `examples/example_implicit_solvent.py`, `example_explicit_solvent.py`, and
`example_protein_ligand.py`. To generate ligand parameters first, see
`rst-parameterize-ligand`.
