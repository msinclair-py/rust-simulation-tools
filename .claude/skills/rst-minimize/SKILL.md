---
name: rst-minimize
description: Energy-minimize an AMBER system with rust_simulation_tools — steepest-descent + conjugate-gradient minimization with optional positional restraints, configurable cycles/cutoff/convergence, and a full energy-component breakdown. Use for "minimize", "energy minimization", "relax the structure", "restrained minimization", or "minimize prmtop/inpcrd".
---

# Energy minimization

Minimizes a system given a prmtop + inpcrd, writing relaxed coordinates and
returning energies/convergence info. Uses steepest-descent for the first
`sd_cycles`, then conjugate gradient.

## Basic usage

```python
from rust_simulation_tools import minimize, minimize_topology, MinimizeConfig

result = minimize(
    "system.prmtop",
    "system.inpcrd",
    output="minimized.inpcrd",   # written if provided
)
result.final_energy   # kcal/mol
result.final_rms      # kcal/(mol*A)
result.cycles
result.converged      # bool
```

## Configuration

```python
config = MinimizeConfig(
    max_cycles=5000,         # total optimization steps
    sd_cycles=100,           # initial steepest-descent steps
    convergence_rms=0.01,    # stop when RMS gradient drops below this
    cutoff=10.0,             # nonbonded cutoff (A)
    restraint_mask="backbone",   # optional selection to restrain (omit for none)
    restraint_weight=10.0,       # kcal/(mol*A^2)
)
result = minimize("system.prmtop", "system.inpcrd", config=config, output="min.inpcrd")
```

## Energy component breakdown

```python
ec = result.energy_components
ec.bond, ec.angle, ec.dihedral
ec.vdw, ec.elec_direct, ec.elec_recip     # PME reciprocal-space term
ec.vdw_14, ec.elec_14                      # 1-4 scaled terms
ec.total()
```

## Reusing a loaded topology

When the topology is already in memory (or you minimize repeatedly), use
`minimize_topology` to skip re-parsing the prmtop:

```python
topo = rst.read_prmtop("system.prmtop")
result = minimize_topology(topo, "system.inpcrd", config=config, output="min.inpcrd")
```

## Typical two-stage protocol

1. Restrained minimization (`restraint_mask="backbone"`, weight ~10) so solvent
   and sidechains relax around a fixed backbone.
2. Unrestrained minimization starting from stage-1 output, with a tighter
   `convergence_rms`.

Full example: `examples/example_minimization.py`.
