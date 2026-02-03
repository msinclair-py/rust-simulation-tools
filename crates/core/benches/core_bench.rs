//! Criterion benchmarks for rst-core functionality.
//!
//! Covers:
//! - Kabsch alignment (single frame, trajectory)
//! - SASA calculation (KD-tree, Shrake-Rupley)
//! - Trajectory unwrapping
//!
//! Run with: cargo bench -p rst-core
//! Run specific group: cargo bench -p rst-core -- kabsch

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rst_core::kabsch::{compute_centroid, compute_kabsch_rotation, kabsch_align};
use rst_core::sasa::{generate_fibonacci_sphere, SASAEngine};
use rst_core::wrapping::unwrap_system;

// ============================================================================
// Synthetic Data Generation
// ============================================================================

/// Generate synthetic atomic coordinates arranged in a helix.
fn generate_helix_coords(n_atoms: usize) -> Vec<[f64; 3]> {
    (0..n_atoms)
        .map(|i| {
            let t = i as f64;
            let angle = t * 0.4;
            let radius = 5.0 + 0.5 * (t * 0.1).sin();
            [radius * angle.cos(), radius * angle.sin(), t * 1.52]
        })
        .collect()
}

/// Generate a synthetic trajectory with slight perturbations from reference.
fn generate_trajectory(n_frames: usize, n_atoms: usize) -> Vec<Vec<[f64; 3]>> {
    let reference = generate_helix_coords(n_atoms);
    (0..n_frames)
        .map(|frame_idx| {
            let phase = frame_idx as f64 * 0.1;
            reference
                .iter()
                .enumerate()
                .map(|(i, coord)| {
                    // Add small perturbations that vary per frame
                    let offset = 0.2 * ((i as f64 + phase).sin());
                    [
                        coord[0] + offset,
                        coord[1] + offset * 0.5,
                        coord[2] + offset * 0.3,
                    ]
                })
                .collect()
        })
        .collect()
}

/// Generate trajectory with periodic wrapping artifacts.
fn generate_wrapped_trajectory(
    n_frames: usize,
    n_atoms: usize,
    box_size: f64,
) -> (Vec<Vec<[f64; 3]>>, Vec<[f64; 3]>) {
    let mut trajectory = Vec::with_capacity(n_frames);
    let box_dimensions = vec![[box_size, box_size, box_size]; n_frames];

    // Initial positions
    let mut current: Vec<[f64; 3]> = (0..n_atoms)
        .map(|i| {
            let row = i / 10;
            let col = i % 10;
            [
                (col as f64) * 3.0 + box_size * 0.1,
                (row as f64) * 3.0 + box_size * 0.1,
                0.0,
            ]
        })
        .collect();

    for _frame_idx in 0..n_frames {
        // Move atoms and wrap periodically
        let frame: Vec<[f64; 3]> = current
            .iter()
            .enumerate()
            .map(|(i, pos)| {
                // Move in x direction, some atoms cross boundary
                let new_x = (pos[0] + 0.5) % box_size;
                let new_y = (pos[1] + 0.1 * (i as f64).sin()) % box_size;
                let new_z = pos[2];
                [new_x, new_y, new_z]
            })
            .collect();

        // Update current positions before wrapping for next iteration
        current = frame
            .iter()
            .map(|pos| [pos[0] + 0.5, pos[1], pos[2]])
            .collect();

        trajectory.push(frame);
    }

    (trajectory, box_dimensions)
}

/// Generate atomic radii for SASA calculations.
fn generate_radii(n_atoms: usize) -> Vec<f64> {
    (0..n_atoms)
        .map(|i| {
            // Mix of carbon, nitrogen, oxygen-like radii
            match i % 4 {
                0 => 1.70, // C
                1 => 1.55, // N
                2 => 1.52, // O
                _ => 1.20, // H
            }
        })
        .collect()
}

/// Generate residue indices for SASA calculations.
fn generate_residue_indices(n_atoms: usize, atoms_per_residue: usize) -> Vec<usize> {
    (0..n_atoms).map(|i| i / atoms_per_residue).collect()
}

// ============================================================================
// Kabsch Alignment Benchmarks
// ============================================================================

fn bench_kabsch(c: &mut Criterion) {
    let mut group = c.benchmark_group("kabsch");

    // Single rotation computation
    for n_align in [10, 50, 100, 500] {
        let mobile = generate_helix_coords(n_align);
        let reference = generate_helix_coords(n_align);
        let mobile_centroid = compute_centroid(&mobile);
        let ref_centroid = compute_centroid(&reference);
        let ref_centered: Vec<[f64; 3]> = reference
            .iter()
            .map(|p| {
                [
                    p[0] - ref_centroid[0],
                    p[1] - ref_centroid[1],
                    p[2] - ref_centroid[2],
                ]
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("rotation_only", n_align),
            &(&mobile, &mobile_centroid, &ref_centered),
            |b, &(mobile, centroid, ref_centered)| {
                b.iter(|| {
                    compute_kabsch_rotation(
                        black_box(mobile),
                        black_box(centroid),
                        black_box(ref_centered),
                    )
                });
            },
        );
    }

    // Full trajectory alignment
    for (n_frames, n_atoms) in [(10, 100), (50, 100), (100, 100), (100, 500), (100, 1000)] {
        let trajectory = generate_trajectory(n_frames, n_atoms);
        let reference = generate_helix_coords(n_atoms);
        // Use subset for alignment (like CA atoms)
        let align_indices: Vec<usize> = (0..n_atoms).step_by(10).collect();

        let label = format!("{}frames_{}atoms", n_frames, n_atoms);
        group.throughput(Throughput::Elements(n_frames as u64));

        group.bench_with_input(
            BenchmarkId::new("trajectory", &label),
            &(&trajectory, &reference, &align_indices),
            |b, &(traj, ref_coords, indices)| {
                b.iter(|| kabsch_align(black_box(traj), black_box(ref_coords), black_box(indices)));
            },
        );
    }

    group.finish();
}

// ============================================================================
// SASA Benchmarks
// ============================================================================

fn bench_sasa(c: &mut Criterion) {
    let mut group = c.benchmark_group("sasa");

    // Fibonacci sphere generation
    for n_points in [92, 162, 642, 960, 2000] {
        group.bench_with_input(
            BenchmarkId::new("fibonacci_sphere", n_points),
            &n_points,
            |b, &n| {
                b.iter(|| generate_fibonacci_sphere(black_box(n)));
            },
        );
    }

    // Engine construction (includes KD-tree build)
    for n_atoms in [100, 300, 1000, 3000] {
        let coords = generate_helix_coords(n_atoms);
        let radii = generate_radii(n_atoms);
        let residue_indices = generate_residue_indices(n_atoms, 10);

        group.bench_with_input(
            BenchmarkId::new("engine_construction", n_atoms),
            &(&coords, &radii, &residue_indices),
            |b, &(coords, radii, res_idx)| {
                b.iter(|| {
                    SASAEngine::new(
                        black_box(coords),
                        black_box(radii),
                        black_box(res_idx),
                        1.4,
                        960,
                    )
                });
            },
        );
    }

    // Full SASA calculation
    for n_atoms in [100, 300, 1000] {
        let coords = generate_helix_coords(n_atoms);
        let radii = generate_radii(n_atoms);
        let residue_indices = generate_residue_indices(n_atoms, 10);
        let engine = SASAEngine::new(&coords, &radii, &residue_indices, 1.4, 960);

        group.throughput(Throughput::Elements(n_atoms as u64));

        group.bench_with_input(
            BenchmarkId::new("per_atom_sasa", n_atoms),
            &engine,
            |b, engine| {
                b.iter(|| engine.calculate_per_atom_sasa());
            },
        );

        group.bench_with_input(
            BenchmarkId::new("per_residue_sasa", n_atoms),
            &engine,
            |b, engine| {
                b.iter(|| engine.calculate_per_residue_sasa());
            },
        );
    }

    // Effect of sphere point count on SASA calculation time
    {
        let n_atoms = 300;
        let coords = generate_helix_coords(n_atoms);
        let radii = generate_radii(n_atoms);
        let residue_indices = generate_residue_indices(n_atoms, 10);

        for n_points in [92, 162, 642, 960] {
            let engine = SASAEngine::new(&coords, &radii, &residue_indices, 1.4, n_points);
            let label = format!("{}pts", n_points);

            group.bench_with_input(
                BenchmarkId::new("varying_precision", &label),
                &engine,
                |b, engine| {
                    b.iter(|| engine.calculate_per_atom_sasa());
                },
            );
        }
    }

    group.finish();
}

// ============================================================================
// Unwrapping Benchmarks
// ============================================================================

fn bench_unwrap(c: &mut Criterion) {
    let mut group = c.benchmark_group("unwrap");

    for (n_frames, n_atoms) in [(50, 100), (100, 100), (100, 500), (100, 1000), (500, 1000)] {
        let (trajectory, box_dimensions) = generate_wrapped_trajectory(n_frames, n_atoms, 50.0);
        let label = format!("{}frames_{}atoms", n_frames, n_atoms);

        group.throughput(Throughput::Elements((n_frames * n_atoms) as u64));

        group.bench_with_input(
            BenchmarkId::new("unwrap_system", &label),
            &(&trajectory, &box_dimensions),
            |b, &(traj, box_dims)| {
                b.iter(|| unwrap_system(black_box(traj), black_box(box_dims)));
            },
        );
    }

    group.finish();
}

// ============================================================================
// Scaling Analysis
// ============================================================================

fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling");
    group.sample_size(20);

    // SASA scaling (expected O(n) with KD-tree optimization)
    for n_atoms in [100, 200, 400, 800, 1600] {
        let coords = generate_helix_coords(n_atoms);
        let radii = generate_radii(n_atoms);
        let residue_indices = generate_residue_indices(n_atoms, 10);
        let engine = SASAEngine::new(&coords, &radii, &residue_indices, 1.4, 960);

        group.throughput(Throughput::Elements(n_atoms as u64));

        group.bench_with_input(
            BenchmarkId::new("sasa_scaling", n_atoms),
            &engine,
            |b, engine| {
                b.iter(|| engine.calculate_per_atom_sasa());
            },
        );
    }

    // Kabsch alignment scaling (fixed frames, varying atoms)
    for n_atoms in [100, 200, 400, 800, 1600] {
        let n_frames = 50;
        let trajectory = generate_trajectory(n_frames, n_atoms);
        let reference = generate_helix_coords(n_atoms);
        let align_indices: Vec<usize> = (0..n_atoms).step_by(10).collect();

        group.throughput(Throughput::Elements(n_atoms as u64));

        group.bench_with_input(
            BenchmarkId::new("kabsch_atom_scaling", n_atoms),
            &(&trajectory, &reference, &align_indices),
            |b, &(traj, ref_coords, indices)| {
                b.iter(|| kabsch_align(black_box(traj), black_box(ref_coords), black_box(indices)));
            },
        );
    }

    // Kabsch alignment scaling (fixed atoms, varying frames)
    for n_frames in [10, 50, 100, 200, 500] {
        let n_atoms = 500;
        let trajectory = generate_trajectory(n_frames, n_atoms);
        let reference = generate_helix_coords(n_atoms);
        let align_indices: Vec<usize> = (0..n_atoms).step_by(10).collect();

        group.throughput(Throughput::Elements(n_frames as u64));

        group.bench_with_input(
            BenchmarkId::new("kabsch_frame_scaling", n_frames),
            &(&trajectory, &reference, &align_indices),
            |b, &(traj, ref_coords, indices)| {
                b.iter(|| kabsch_align(black_box(traj), black_box(ref_coords), black_box(indices)));
            },
        );
    }

    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    benches,
    bench_kabsch,
    bench_sasa,
    bench_unwrap,
    bench_scaling,
);
criterion_main!(benches);
