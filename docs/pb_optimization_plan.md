# PB Solver Performance Optimization Plan

## Current Performance Baseline

- **~48s per frame** for MM-PBSA binding energy (6 PB solves: 2 per subsystem x 3 subsystems)
- **~0.1s per frame** for MM-GBSA (for reference)
- Test system: 5846 solute atoms (307 receptor residues + 59 ligand residues), 64421 total atoms (solvated)
- Grid spacing: 0.5 A, buffer: 10.0 A

---

## 1. Parallelize SOR Red-Black Sweeps (rayon)

**Impact: 4-8x speedup | Effort: Low**

The SOR solver in `pb_solver.rs` iterates over all interior grid points sequentially.
The red-black ordering already partitions points into two independent sets per iteration —
all points of the same color can be updated simultaneously without data races.

### Current code (`pb_solver.rs:143-192`)

```rust
for color in 0..2 {
    for iz in 1..nz-1 {
        for iy in 1..ny-1 {
            for ix in 1..nx-1 {
                if (ix + iy + iz) % 2 != color { continue; }
                // update potential[idx] ...
            }
        }
    }
}
```

### Proposed change

Replace the `iz` loop with `rayon::iter::ParallelIterator` over z-slices.
Each z-slice writes to disjoint grid indices (same-color points don't neighbor
each other), so no synchronization is needed beyond a parallel reduction for
the RMS residual.

```rust
use rayon::prelude::*;

for color in 0..2 {
    let slice_residuals: f64 = (1..nz-1).into_par_iter().map(|iz| {
        let mut local_rms = 0.0;
        let mut local_count = 0u64;
        for iy in 1..ny-1 {
            for ix in 1..nx-1 {
                if (ix + iy + iz) % 2 != color { continue; }
                // ... same update logic ...
                local_rms += residual * residual;
                local_count += 1;
            }
        }
        (local_rms, local_count)
    }).reduce(|| (0.0, 0), |(a, b), (c, d)| (a + c, b + d));
    // accumulate into rms_residual / count
}
```

This requires the `potential` array to be shared mutably across threads.
Since same-color points never neighbor each other on the same color pass,
this is safe to do with `UnsafeCell` or by splitting the array into
non-overlapping slices per z-plane. Alternatively, use `AtomicF64` or
`unsafe` with a documented safety invariant (standard practice for
red-black SOR).

### Why it works

Red-black SOR guarantees that during a single color sweep, no two updated
points are adjacent. The 7-point stencil only reads neighbors of the
opposite color, which are not being written in the current sweep. This is
the same parallelization strategy used by APBS and other production PB codes.

---

## 2. Eliminate Potential Vector Clone in Energy Calculation

**Impact: ~1.1-1.3x speedup, eliminates ~4MB allocation per solve | Effort: Trivial**

`compute_elec_energy()` in `pb_solver.rs` creates a full copy of the
potential array solely to call `PbGrid::interpolate()`:

```rust
pub fn compute_elec_energy(grid: &PbGrid, potential: &[f64], ...) -> f64 {
    let pot_grid = PbGrid {
        dims: grid.dims,
        spacing: grid.spacing,
        origin: grid.origin,
        data: potential.to_vec(),  // <-- full clone of ~1M f64s
    };
    // ... pot_grid.interpolate(c) ...
}
```

### Proposed change

Add an `interpolate_from()` method to `PbGrid` that accepts an external
data slice, or refactor `interpolate()` to take `&[f64]` as a parameter
instead of reading from `self.data`. This avoids the allocation entirely.

```rust
pub fn interpolate_with_data(&self, point: &[f64; 3], data: &[f64]) -> f64 {
    // same trilinear interpolation logic, reads from `data` instead of `self.data`
}
```

Then `compute_elec_energy` becomes zero-allocation:

```rust
pub fn compute_elec_energy(grid: &PbGrid, potential: &[f64], ...) -> f64 {
    let mut energy = 0.0;
    for (c, &q) in coords.iter().zip(charges.iter()) {
        let phi = grid.interpolate_with_data(c, potential);
        energy += q * phi;
    }
    0.5 * energy
}
```

For a typical 80^3 grid, this saves 4MB per call (called 2x per
`compute_pb_energy`, 6x per binding frame, thousands of times per
trajectory).

---

## 3. Parallelize Solvated + Reference Solves

**Impact: ~1.8x speedup per `compute_pb_energy` call | Effort: Low-Medium**

Each call to `compute_pb_energy()` runs two independent PB solves
sequentially:

1. **Solvated solve**: epsilon_in/epsilon_out dielectrics, ionic strength
2. **Reference solve**: uniform epsilon_in, no ionic strength

These share only read-only data (grid geometry, charge map, coords) and
write to separate output buffers.

### Proposed change

Use `rayon::join()` to run both solves concurrently:

```rust
let (result_solv, result_ref) = rayon::join(
    || solve_lpbe(&grid, &charge_map, &diel_solv, &kappa_sq_solv, ...),
    || solve_lpbe(&grid, &charge_map, &diel_ref, &kappa_sq_ref, ...),
);
```

The dielectric maps and kappa arrays must be constructed before the join,
but they are also independent and can be parallelized.

### Caveat

If optimization #1 (parallel SOR) is already saturating all cores, this
won't help much. The two optimizations compose best when inner SOR
parallelism uses a bounded thread pool, leaving cores available for the
outer parallelism. In practice, `rayon`'s work-stealing scheduler handles
nested parallelism well.

---

## 4. Parallelize Complex / Receptor / Ligand Subsystem Solves

**Impact: ~2-3x speedup for binding energy | Effort: Medium**

The binding energy workflow in `binding.rs` computes energies for three
subsystems sequentially:

```
complex  = MM + polar + SA   (2 PB solves)
receptor = MM + polar + SA   (2 PB solves)
ligand   = MM + polar + SA   (2 PB solves)
```

All three subsystems are independent — they operate on different
sub-topologies and coordinate subsets.

### Proposed change

Wrap the three `compute_subsystem_energy()` calls in `rayon::join()` or
`rayon::scope()`:

```rust
let (complex, receptor, ligand) = rayon::join3(
    || compute_subsystem_energy(&complex_top, &complex_coords, ...),
    || compute_subsystem_energy(&receptor_top, &receptor_coords, ...),
    || compute_subsystem_energy(&ligand_top, &ligand_coords, ...),
);
```

Note: `rayon::join` only supports 2-way parallelism, so use nested joins
or `rayon::scope` for 3-way.

### Interaction with other optimizations

If the SOR solver is already parallel (optimization #1), the subsystem
solves will compete for the same thread pool. The optimal strategy depends
on system size:

- **Large systems** (>5000 atoms): Inner SOR parallelism dominates; subsystem
  parallelism adds little.
- **Small systems** (<1000 atoms): SOR has less work to parallelize; subsystem
  parallelism is more valuable.

A practical approach: parallelize subsystems and let rayon's work-stealing
scheduler balance the load automatically.

---

## 5. Adaptive Grid Sizing per Subsystem

**Impact: 2-5x overall speedup | Effort: Medium-High**

Currently, `compute_pb_energy()` calls `auto_grid(coords, spacing, buffer)`
which sizes the grid to fit all provided coordinates plus a buffer. This is
correct, but in the binding workflow each subsystem gets its own grid:

| Subsystem | Typical atoms | Grid size (0.5A) | Grid points |
|-----------|--------------|-------------------|-------------|
| Complex   | 5846         | ~120x110x100      | 1.3M        |
| Receptor  | 4900         | ~115x105x95       | 1.1M        |
| Ligand    | 946          | ~55x50x45         | 124K        |

The ligand grid is **~10x smaller** than the complex grid. Since SOR
convergence scales with grid size, the ligand solve is already ~10x cheaper
than the complex solve — but only if the grid is sized to the ligand.

### Current behavior

This already works correctly because `compute_pb_energy` calls `auto_grid`
on the subsystem coordinates, which are sliced by `extract_coords`. The
grid automatically adapts to each subsystem's bounding box.

### Verification needed

Confirm that the receptor and ligand PB solves are indeed using smaller
grids by adding logging:

```rust
log::info!("PB grid: {}x{}x{} ({} points) for {} atoms",
    grid.dims[0], grid.dims[1], grid.dims[2], grid.len(), coords.len());
```

If this is already the case, no code change is needed — just verification
that the existing implementation is optimal in this regard.

---

## 6. Cache-Friendly Tiling in SOR Loop

**Impact: 1.5-2.5x speedup | Effort: Medium**

The SOR stencil accesses 6 neighbors per grid point. In the z-direction,
neighbors are `nx * ny` elements apart:

```
potential[idx + 1]          // x+1: stride 1 (cache-friendly)
potential[idx - 1]          // x-1: stride 1
potential[idx + nx]         // y+1: stride nx (~120)
potential[idx - nx]         // y-1: stride -nx
potential[idx + nx*ny]      // z+1: stride nx*ny (~13,200)
potential[idx - nx*ny]      // z-1: stride -nx*ny
```

For a 120x110x100 grid, the z-direction stride is 13,200 elements =
105.6 KB. This exceeds L1 cache (typically 32-48 KB) and L2 for some
architectures, causing frequent cache misses.

### Proposed change: 3D loop tiling

Process the grid in small 3D tiles that fit in L1/L2 cache:

```rust
const TILE: usize = 16; // Tune for target architecture

for tz in (1..nz-1).step_by(TILE) {
    for ty in (1..ny-1).step_by(TILE) {
        for tx in (1..nx-1).step_by(TILE) {
            let iz_end = (tz + TILE).min(nz - 1);
            let iy_end = (ty + TILE).min(ny - 1);
            let ix_end = (tx + TILE).min(nx - 1);
            for iz in tz..iz_end {
                for iy in ty..iy_end {
                    for ix in tx..ix_end {
                        if (ix + iy + iz) % 2 != color { continue; }
                        // ... update ...
                    }
                }
            }
        }
    }
}
```

A 16x16x16 tile uses `16*16*16 * 8 bytes = 32 KB` for the potential
values, fitting neatly in L1 cache. The neighbor accesses for z-direction
now span at most `16 * nx * 8 bytes` which is more cache-friendly.

### Caveat

Tiling interacts with red-black ordering and parallelization. The tile
boundaries must respect the coloring constraint. This is solvable but
adds implementation complexity.

---

## 7. Multigrid Acceleration

**Impact: 2-10x speedup | Effort: High**

The single-level SOR solver converges slowly for large grids. The number
of iterations needed scales as O(N^{2/3}) where N is the number of grid
points. For a 1M-point grid, this means ~10,000 iterations (our current
default max).

Multigrid methods solve the PB equation on a hierarchy of grids (coarse
to fine), using coarse-grid solutions to accelerate fine-grid convergence.
This reduces the iteration count to O(1) per multigrid cycle, with total
work proportional to the number of grid points.

### V-cycle multigrid sketch

```
Level 0 (finest):  120x110x100  — pre-smooth (3 SOR iters)
  → restrict residual to Level 1
Level 1:            60x55x50    — pre-smooth (3 SOR iters)
  → restrict to Level 2
Level 2:            30x28x25   — solve exactly or smooth heavily
  ← prolongate correction to Level 1
Level 1:                        — post-smooth (3 SOR iters)
  ← prolongate correction to Level 0
Level 0:                        — post-smooth (3 SOR iters)
```

### Required components

1. **Restriction operator**: Transfer residuals from fine to coarse grid
   (full-weighting or half-weighting)
2. **Prolongation operator**: Interpolate corrections from coarse to fine
   grid (trilinear)
3. **Coarse-grid operator**: Re-discretize the PBE on coarser grids
   (requires coarse dielectric/kappa maps)
4. **V-cycle or F-cycle driver**: Orchestrate the multi-level solve

### Complexity estimate

- ~400-600 lines of new Rust code
- Requires coarse-grid versions of dielectric and kappa maps
- Boundary condition handling at each level
- Testing against single-level solver for correctness

### Alternative: Preconditioned Conjugate Gradient

A simpler algorithmic improvement is to replace SOR with a preconditioned
conjugate gradient (PCG) solver using an incomplete Cholesky or Jacobi
preconditioner. This typically converges 2-3x faster than SOR with less
implementation effort than multigrid.

---

## Summary

| # | Optimization | Speedup | Effort | Dependencies |
|---|-------------|---------|--------|--------------|
| 1 | Parallel SOR (rayon) | 4-8x | Low | None |
| 2 | Eliminate potential clone | 1.1-1.3x | Trivial | None |
| 3 | Parallel solvated+reference solves | ~1.8x | Low | None |
| 4 | Parallel subsystem solves | 2-3x | Medium | None |
| 5 | Adaptive grid sizing (verify) | 2-5x | Low | None |
| 6 | Cache-friendly tiling | 1.5-2.5x | Medium | Interacts with #1 |
| 7 | Multigrid / PCG solver | 2-10x | High | None |

### Recommended implementation order

1. **Phase 1** (quick wins): #1 + #2 + #3 — Expected combined: ~8-12x
2. **Phase 2** (medium effort): #4 + #5 verification + #6 — Expected additional: ~2-4x
3. **Phase 3** (algorithmic): #7 — Expected additional: ~2-5x

### Projected performance

| Phase | Time per frame | Notes |
|-------|---------------|-------|
| Current | ~48s | Sequential SOR, no parallelism |
| After Phase 1 | ~4-6s | Parallel SOR + parallel solves |
| After Phase 2 | ~1.5-3s | + tiling + subsystem parallelism |
| After Phase 3 | ~0.3-1s | + multigrid solver |
