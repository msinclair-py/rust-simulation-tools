//! Criterion benchmarks for the MM-PBSA energy components.
//!
//! Comprehensive benchmark suite covering:
//! - MM energy calculation (with/without pre-built NB sets)
//! - GB solvation energy (OBC-I and OBC-II models)
//! - SA non-polar energy (Shrake-Rupley)
//! - PB solvation energy (multigrid solver)
//! - Per-residue decomposition
//! - Full binding energy pipeline
//! - Entropy estimation
//! - NB set construction
//!
//! Run with: cargo bench -p rst-mmpbsa
//! Run specific group: cargo bench -p rst-mmpbsa -- mm_energy
//! Run with verbose output: cargo bench -p rst-mmpbsa -- --verbose

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rst_core::amber::prmtop::AmberTopology;
use rst_mmpbsa::binding::{
    BindingConfig, FrameEnergy, SolvationMethod, SubsystemEnergy, TrajectoryFormat,
};
use rst_mmpbsa::decomposition::decompose_binding_energy;
use rst_mmpbsa::entropy::interaction_entropy;
use rst_mmpbsa::gb_energy::{GbModel, GbParams};
use rst_mmpbsa::mm_energy::{compute_mm_energy, compute_mm_energy_with_nb, PairBitmap};
use rst_mmpbsa::pb_energy::{compute_pb_energy, PbParams};
use rst_mmpbsa::sa_energy::{compute_sa_energy, SaParams};
use std::f64::consts::PI;
use std::sync::Arc;

// ============================================================================
// Synthetic Data Generation
// ============================================================================

/// Build a synthetic AmberTopology with `n_residues` residues, each containing
/// `atoms_per_residue` atoms arranged as a simple chain. This produces a
/// topology with bonds, angles, dihedrals, exclusions, charges, LJ parameters,
/// radii, and screening factors -- enough to exercise every energy term.
fn build_synthetic_topology(n_residues: usize, atoms_per_residue: usize) -> AmberTopology {
    let n_atoms = n_residues * atoms_per_residue;
    let n_types = 4; // small number of atom types

    // Per-atom arrays
    let mut atom_names = Vec::with_capacity(n_atoms);
    let mut atom_type_indices = Vec::with_capacity(n_atoms);
    let mut charges = Vec::with_capacity(n_atoms);
    let mut charges_amber = Vec::with_capacity(n_atoms);
    let mut masses = Vec::with_capacity(n_atoms);
    let mut radii = Vec::with_capacity(n_atoms);
    let mut screen = Vec::with_capacity(n_atoms);
    let mut atom_sigmas = Vec::with_capacity(n_atoms);
    let mut atom_epsilons = Vec::with_capacity(n_atoms);
    let mut residue_labels = Vec::with_capacity(n_residues);
    let mut residue_pointers = Vec::with_capacity(n_residues);

    for res in 0..n_residues {
        residue_labels.push("ALA".to_string());
        residue_pointers.push(res * atoms_per_residue);
        for a in 0..atoms_per_residue {
            let idx = res * atoms_per_residue + a;
            atom_names.push(format!("A{}", a));
            atom_type_indices.push(idx % n_types);
            // Alternate positive/negative charges for electrostatics
            let q = if idx % 2 == 0 { 0.1 } else { -0.1 };
            charges.push(q);
            charges_amber.push(q * 18.2223);
            masses.push(12.0);
            radii.push(1.5 + 0.2 * (idx % 3) as f64);
            screen.push(0.72);
            atom_sigmas.push(1.7);
            atom_epsilons.push(0.1);
        }
    }

    // Bonds: connect sequential atoms within each residue
    let mut bonds = Vec::new();
    let mut bond_types = Vec::new();
    for res in 0..n_residues {
        let base = res * atoms_per_residue;
        for a in 0..(atoms_per_residue - 1) {
            bonds.push((base + a, base + a + 1));
            bond_types.push(0);
        }
        // Inter-residue bond (peptide bond analog)
        if res + 1 < n_residues {
            bonds.push((base + atoms_per_residue - 1, (res + 1) * atoms_per_residue));
            bond_types.push(0);
        }
    }

    // Angles: sequential triplets along the chain
    let mut angles = Vec::new();
    for i in 0..(n_atoms.saturating_sub(2)) {
        if i + 2 < n_atoms {
            angles.push((i, i + 1, i + 2, 0));
        }
    }
    // Limit angles to keep benchmark setup fast for large systems
    if angles.len() > n_atoms {
        angles.truncate(n_atoms);
    }

    // Dihedrals: sequential quadruplets
    let mut dihedrals = Vec::new();
    for i in 0..(n_atoms.saturating_sub(3)) {
        if i + 3 < n_atoms {
            dihedrals.push((i, i + 1, i + 2, i + 3, 0, false));
        }
    }
    if dihedrals.len() > n_atoms {
        dihedrals.truncate(n_atoms);
    }

    // Exclusion lists: exclude bonded neighbors (1-2 pairs)
    let mut neighbors: Vec<Vec<usize>> = vec![vec![]; n_atoms];
    for &(a, b) in &bonds {
        neighbors[a].push(b);
        neighbors[b].push(a);
    }
    let mut num_excluded_atoms = Vec::with_capacity(n_atoms);
    let mut excluded_atoms_list = Vec::new();
    for i in 0..n_atoms {
        let mut excl: Vec<usize> = neighbors[i].iter().copied().filter(|&j| j > i).collect();
        excl.sort();
        num_excluded_atoms.push(excl.len());
        excluded_atoms_list.extend_from_slice(&excl);
    }

    // LJ coefficient tables (n_types x n_types, stored as triangular)
    let ntypes2 = n_types * n_types;
    let mut nb_parm_index = vec![0i64; ntypes2];
    let mut lj_acoef = Vec::new();
    let mut lj_bcoef = Vec::new();
    let mut idx: i64 = 1;
    for ti in 0..n_types {
        for tj in 0..n_types {
            nb_parm_index[n_types * ti + tj] = idx;
            lj_acoef.push(1_000_000.0); // repulsive
            lj_bcoef.push(500.0); // attractive
            idx += 1;
        }
    }

    AmberTopology {
        n_atoms,
        n_residues,
        n_types,
        atom_names,
        atom_type_indices,
        charges,
        charges_amber,
        residue_labels,
        residue_pointers,
        lj_sigma: Arc::new(vec![]),
        lj_epsilon: Arc::new(vec![]),
        atom_sigmas,
        atom_epsilons,
        bonds,
        bond_types,
        masses,
        radii,
        screen,
        bond_force_constants: Arc::new(vec![300.0]),
        bond_equil_values: Arc::new(vec![1.52]),
        angle_force_constants: Arc::new(vec![50.0]),
        angle_equil_values: Arc::new(vec![1.911]),
        dihedral_force_constants: Arc::new(vec![2.0]),
        dihedral_periodicities: Arc::new(vec![2.0]),
        dihedral_phases: Arc::new(vec![PI]),
        angles,
        dihedrals,
        num_excluded_atoms,
        excluded_atoms_list,
        scee_scale_factor: 1.2,
        scnb_scale_factor: 2.0,
        lj_acoef: Arc::new(lj_acoef),
        lj_bcoef: Arc::new(lj_bcoef),
        nb_parm_index: Arc::new(nb_parm_index),
    }
}

/// Generate synthetic coordinates arranged in a loose helix so atoms are
/// spatially distributed but not overlapping.
fn build_synthetic_coords(n_atoms: usize) -> Vec<[f64; 3]> {
    let mut coords = Vec::with_capacity(n_atoms);
    for i in 0..n_atoms {
        let t = i as f64;
        // Helix: x = R*cos(t*step), y = R*sin(t*step), z = rise*t
        let angle = t * 0.4; // ~23 degrees per atom
        let radius = 5.0 + 0.5 * (t * 0.1).sin();
        coords.push([
            radius * angle.cos(),
            radius * angle.sin(),
            t * 1.52, // ~bond length rise per atom
        ]);
    }
    coords
}

/// Pre-build exclusion and 1-4 pair sets (mirrors what the binding code does).
fn build_nb_sets(topology: &AmberTopology) -> (PairBitmap, PairBitmap) {
    let excluded = rst_mmpbsa::mm_energy::build_exclusion_set(topology);
    let pairs_14 = rst_mmpbsa::mm_energy::build_14_pairs(topology);
    (excluded, pairs_14)
}

/// Generate synthetic frame energies for entropy benchmarks.
fn build_synthetic_frame_energies(n_frames: usize) -> Vec<FrameEnergy> {
    (0..n_frames)
        .map(|i| {
            let phase = i as f64 * 0.1;
            let delta_mm = -50.0 + 5.0 * phase.sin();
            let delta_polar = 10.0 + 2.0 * (phase * 1.3).cos();
            let delta_sa = -3.0 + 0.5 * (phase * 0.7).sin();
            let delta_total = delta_mm + delta_polar + delta_sa;

            // Create subsystem energies that produce the desired deltas
            let complex = SubsystemEnergy {
                mm: -100.0 + 3.0 * phase.sin(),
                polar: 20.0 + phase.cos(),
                sa: -5.0,
            };
            let receptor = SubsystemEnergy {
                mm: complex.mm - delta_mm * 0.5,
                polar: complex.polar - delta_polar * 0.5,
                sa: complex.sa - delta_sa * 0.5,
            };
            let ligand = SubsystemEnergy {
                mm: complex.mm - receptor.mm - delta_mm,
                polar: complex.polar - receptor.polar - delta_polar,
                sa: complex.sa - receptor.sa - delta_sa,
            };

            FrameEnergy {
                complex,
                receptor,
                ligand,
                delta_mm,
                delta_polar,
                delta_sa,
                delta_total,
            }
        })
        .collect()
}

// ============================================================================
// System Size Configurations
// ============================================================================

/// Standard system sizes for benchmarking
const SYSTEM_SIZES: &[(usize, usize)] = &[
    (10, 10),  // 100 atoms - small
    (30, 10),  // 300 atoms - medium
    (100, 10), // 1000 atoms - large
    (300, 10), // 3000 atoms - very large (optional)
];

/// Smaller sizes for expensive operations (PB, decomposition)
const SMALL_SIZES: &[(usize, usize)] = &[
    (5, 5),   // 25 atoms
    (10, 10), // 100 atoms
    (20, 10), // 200 atoms
];

// ============================================================================
// MM Energy Benchmarks
// ============================================================================

fn bench_mm_energy(c: &mut Criterion) {
    let mut group = c.benchmark_group("mm_energy");

    for &(n_res, atoms_per_res) in SYSTEM_SIZES.iter().take(3) {
        let top = build_synthetic_topology(n_res, atoms_per_res);
        let coords = build_synthetic_coords(top.n_atoms);
        let (excluded, pairs_14) = build_nb_sets(&top);
        let label = format!("{}atoms", top.n_atoms);

        group.throughput(Throughput::Elements(top.n_atoms as u64));

        // With pre-built NB sets (the hot path in production)
        group.bench_with_input(
            BenchmarkId::new("with_prebuilt_nb", &label),
            &(&top, &coords, &excluded, &pairs_14),
            |b, &(top, coords, excl, p14)| {
                b.iter(|| compute_mm_energy_with_nb(black_box(top), black_box(coords), excl, p14));
            },
        );

        // Without pre-built NB sets (includes set construction overhead)
        group.bench_with_input(
            BenchmarkId::new("with_nb_build", &label),
            &(&top, &coords),
            |b, &(top, coords)| {
                b.iter(|| compute_mm_energy(black_box(top), black_box(coords)));
            },
        );
    }
    group.finish();
}

// ============================================================================
// GB Energy Benchmarks
// ============================================================================

fn bench_gb_energy(c: &mut Criterion) {
    let mut group = c.benchmark_group("gb_energy");

    // OBC-I at multiple sizes
    for &(n_res, atoms_per_res) in SYSTEM_SIZES.iter().take(3) {
        let top = build_synthetic_topology(n_res, atoms_per_res);
        let coords = build_synthetic_coords(top.n_atoms);
        let label = format!("{}atoms", top.n_atoms);
        let gb_params = GbParams::default();

        group.throughput(Throughput::Elements(top.n_atoms as u64));

        group.bench_with_input(
            BenchmarkId::new("obc1", &label),
            &(&top, &coords, &gb_params),
            |b, &(top, coords, params)| {
                b.iter(|| {
                    rst_mmpbsa::gb_energy::compute_gb_energy(
                        black_box(top),
                        black_box(coords),
                        params,
                    )
                });
            },
        );
    }

    // OBC-II comparison at medium size
    {
        let top = build_synthetic_topology(30, 10);
        let coords = build_synthetic_coords(top.n_atoms);
        let obc2_params = GbParams {
            model: GbModel::ObcII,
            ..GbParams::default()
        };
        group.bench_with_input(
            BenchmarkId::new("obc2", "300atoms"),
            &(&top, &coords, &obc2_params),
            |b, &(top, coords, params)| {
                b.iter(|| {
                    rst_mmpbsa::gb_energy::compute_gb_energy(
                        black_box(top),
                        black_box(coords),
                        params,
                    )
                });
            },
        );
    }
    group.finish();
}

// ============================================================================
// SA Energy Benchmarks
// ============================================================================

fn bench_sa_energy(c: &mut Criterion) {
    let mut group = c.benchmark_group("sa_energy");
    let sa_params = SaParams::default();

    for &(n_res, atoms_per_res) in SYSTEM_SIZES.iter().take(3) {
        let top = build_synthetic_topology(n_res, atoms_per_res);
        let coords = build_synthetic_coords(top.n_atoms);
        let label = format!("{}atoms", top.n_atoms);

        group.throughput(Throughput::Elements(top.n_atoms as u64));

        group.bench_with_input(
            BenchmarkId::new("shrake_rupley", &label),
            &(&top, &coords, &sa_params),
            |b, &(top, coords, params)| {
                b.iter(|| compute_sa_energy(black_box(top), black_box(coords), params));
            },
        );
    }

    // Benchmark with different sphere point counts
    {
        let top = build_synthetic_topology(30, 10);
        let coords = build_synthetic_coords(top.n_atoms);

        for n_points in [92, 162, 642, 960] {
            let params = SaParams {
                n_sphere_points: n_points,
                ..SaParams::default()
            };
            let label = format!("{}pts_300atoms", n_points);

            group.bench_with_input(
                BenchmarkId::new("varying_precision", &label),
                &(&top, &coords, &params),
                |b, &(top, coords, params)| {
                    b.iter(|| compute_sa_energy(black_box(top), black_box(coords), params));
                },
            );
        }
    }
    group.finish();
}

// ============================================================================
// PB Energy Benchmarks
// ============================================================================

fn bench_pb_energy(c: &mut Criterion) {
    let mut group = c.benchmark_group("pb_energy");
    // PB is expensive — use small systems and limit sample size
    group.sample_size(10);

    for &(n_res, atoms_per_res) in SMALL_SIZES.iter().take(2) {
        let top = build_synthetic_topology(n_res, atoms_per_res);
        let coords = build_synthetic_coords(top.n_atoms);
        let label = format!("{}atoms", top.n_atoms);

        let params = PbParams {
            grid_spacing: 0.5,
            grid_buffer: 10.0,
            tolerance: 1e-6,
            max_iterations: 10000,
            ..PbParams::default()
        };

        group.bench_with_input(
            BenchmarkId::new("multigrid", &label),
            &(&top, &coords, &params),
            |b, &(top, coords, params)| {
                b.iter(|| compute_pb_energy(black_box(top), black_box(coords), params));
            },
        );
    }

    // Grid spacing comparison
    {
        let top = build_synthetic_topology(10, 10);
        let coords = build_synthetic_coords(top.n_atoms);

        for spacing in [1.0, 0.5, 0.25] {
            let params = PbParams {
                grid_spacing: spacing,
                grid_buffer: 10.0,
                tolerance: 1e-6,
                max_iterations: 10000,
                ..PbParams::default()
            };
            let label = format!("{:.2}A_100atoms", spacing);

            group.bench_with_input(
                BenchmarkId::new("grid_spacing", &label),
                &(&top, &coords, &params),
                |b, &(top, coords, params)| {
                    b.iter(|| compute_pb_energy(black_box(top), black_box(coords), params));
                },
            );
        }
    }
    group.finish();
}

// ============================================================================
// Full Binding Energy Pipeline Benchmarks
// ============================================================================

fn bench_binding_single_frame(c: &mut Criterion) {
    let mut group = c.benchmark_group("binding_single_frame");

    for &(n_receptor_res, n_ligand_res, atoms_per_res) in &[(20, 10, 10), (50, 25, 10)] {
        let total_res = n_receptor_res + n_ligand_res;
        let top = build_synthetic_topology(total_res, atoms_per_res);
        let coords = build_synthetic_coords(top.n_atoms);
        let label = format!("{}atoms_{}r_{}l", top.n_atoms, n_receptor_res, n_ligand_res);

        let config = BindingConfig {
            receptor_residues: (0..n_receptor_res).collect(),
            ligand_residues: (n_receptor_res..total_res).collect(),
            solvation_method: SolvationMethod::GB(GbParams::default()),
            sa_params: SaParams::default(),
            trajectory_format: TrajectoryFormat::Mdcrd { has_box: false },
            stride: 1,
            start_frame: 0,
            end_frame: usize::MAX,
        };

        group.bench_with_input(
            BenchmarkId::new("gb_pipeline", &label),
            &(&top, &coords, &config),
            |b, &(top, coords, config)| {
                b.iter(|| {
                    rst_mmpbsa::binding::compute_binding_energy_single_frame(
                        black_box(top),
                        black_box(coords),
                        config,
                    )
                    .unwrap()
                });
            },
        );
    }
    group.finish();
}

fn bench_binding_single_frame_pb(c: &mut Criterion) {
    let mut group = c.benchmark_group("binding_single_frame_pb");
    group.sample_size(10);

    let n_receptor_res = 10;
    let n_ligand_res = 5;
    let atoms_per_res = 5;
    let total_res = n_receptor_res + n_ligand_res;
    let top = build_synthetic_topology(total_res, atoms_per_res);
    let coords = build_synthetic_coords(top.n_atoms);
    let label = format!("{}atoms_{}r_{}l", top.n_atoms, n_receptor_res, n_ligand_res);

    let config = BindingConfig {
        receptor_residues: (0..n_receptor_res).collect(),
        ligand_residues: (n_receptor_res..total_res).collect(),
        solvation_method: SolvationMethod::PB(PbParams {
            grid_spacing: 0.5,
            grid_buffer: 10.0,
            tolerance: 1e-6,
            max_iterations: 10000,
            ..PbParams::default()
        }),
        sa_params: SaParams::default(),
        trajectory_format: TrajectoryFormat::Mdcrd { has_box: false },
        stride: 1,
        start_frame: 0,
        end_frame: usize::MAX,
    };

    group.bench_with_input(
        BenchmarkId::new("pb_pipeline", &label),
        &(&top, &coords, &config),
        |b, &(top, coords, config)| {
            b.iter(|| {
                rst_mmpbsa::binding::compute_binding_energy_single_frame(
                    black_box(top),
                    black_box(coords),
                    config,
                )
                .unwrap()
            });
        },
    );
    group.finish();
}

// ============================================================================
// Per-Residue Decomposition Benchmarks
// ============================================================================

fn bench_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("decomposition");
    group.sample_size(20);

    for &(n_receptor_res, n_ligand_res, atoms_per_res) in &[(15, 5, 8), (30, 10, 8)] {
        let total_res = n_receptor_res + n_ligand_res;
        let top = build_synthetic_topology(total_res, atoms_per_res);
        let coords = build_synthetic_coords(top.n_atoms);
        let label = format!("{}atoms_{}res", top.n_atoms, n_receptor_res + n_ligand_res);

        let receptor_residues: Vec<usize> = (0..n_receptor_res).collect();
        let ligand_residues: Vec<usize> = (n_receptor_res..total_res).collect();
        let gb_params = GbParams::default();
        let sa_params = SaParams::default();

        group.throughput(Throughput::Elements(total_res as u64));

        group.bench_with_input(
            BenchmarkId::new("per_residue", &label),
            &(
                &top,
                &coords,
                &receptor_residues,
                &ligand_residues,
                &gb_params,
                &sa_params,
            ),
            |b, &(top, coords, rec, lig, gb, sa)| {
                b.iter(|| {
                    decompose_binding_energy(
                        black_box(top),
                        black_box(coords),
                        black_box(rec),
                        black_box(lig),
                        gb,
                        sa,
                    )
                    .unwrap()
                });
            },
        );
    }
    group.finish();
}

// ============================================================================
// Entropy Estimation Benchmarks
// ============================================================================

fn bench_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy");

    for n_frames in [100, 500, 1000, 5000] {
        let frames = build_synthetic_frame_energies(n_frames);
        let label = format!("{}frames", n_frames);

        group.throughput(Throughput::Elements(n_frames as u64));

        group.bench_with_input(
            BenchmarkId::new("interaction_entropy", &label),
            &frames,
            |b, frames| {
                b.iter(|| interaction_entropy(black_box(frames), black_box(298.15)));
            },
        );
    }
    group.finish();
}

// ============================================================================
// NB Set Construction Benchmarks
// ============================================================================

fn bench_nb_set_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("nb_set_construction");

    for &(n_res, atoms_per_res) in SYSTEM_SIZES.iter().take(3) {
        let top = build_synthetic_topology(n_res, atoms_per_res);
        let label = format!("{}atoms", top.n_atoms);

        group.bench_with_input(BenchmarkId::new("exclusion_set", &label), &top, |b, top| {
            b.iter(|| rst_mmpbsa::mm_energy::build_exclusion_set(black_box(top)));
        });

        group.bench_with_input(BenchmarkId::new("14_pairs", &label), &top, |b, top| {
            b.iter(|| rst_mmpbsa::mm_energy::build_14_pairs(black_box(top)));
        });
    }
    group.finish();
}

// ============================================================================
// Scaling Analysis Benchmarks
// ============================================================================

/// Benchmark to analyze scaling behavior across system sizes.
fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling_analysis");
    group.sample_size(20);

    // Test GB energy scaling (expected O(n^2))
    for n_atoms in [50, 100, 200, 400, 800] {
        let n_res = n_atoms / 10;
        let atoms_per_res = 10;
        let top = build_synthetic_topology(n_res, atoms_per_res);
        let coords = build_synthetic_coords(top.n_atoms);
        let gb_params = GbParams::default();

        group.throughput(Throughput::Elements(top.n_atoms as u64));

        group.bench_with_input(
            BenchmarkId::new("gb_scaling", n_atoms),
            &(&top, &coords, &gb_params),
            |b, &(top, coords, params)| {
                b.iter(|| {
                    rst_mmpbsa::gb_energy::compute_gb_energy(
                        black_box(top),
                        black_box(coords),
                        params,
                    )
                });
            },
        );
    }

    // Test MM energy scaling
    for n_atoms in [50, 100, 200, 400, 800] {
        let n_res = n_atoms / 10;
        let atoms_per_res = 10;
        let top = build_synthetic_topology(n_res, atoms_per_res);
        let coords = build_synthetic_coords(top.n_atoms);
        let (excluded, pairs_14) = build_nb_sets(&top);

        group.throughput(Throughput::Elements(top.n_atoms as u64));

        group.bench_with_input(
            BenchmarkId::new("mm_scaling", n_atoms),
            &(&top, &coords, &excluded, &pairs_14),
            |b, &(top, coords, excl, p14)| {
                b.iter(|| compute_mm_energy_with_nb(black_box(top), black_box(coords), excl, p14));
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
    bench_mm_energy,
    bench_gb_energy,
    bench_sa_energy,
    bench_binding_single_frame,
    bench_nb_set_construction,
    bench_pb_energy,
    bench_binding_single_frame_pb,
    bench_decomposition,
    bench_entropy,
    bench_scaling,
);
criterion_main!(benches);
