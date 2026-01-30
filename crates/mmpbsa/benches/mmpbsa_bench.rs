//! Criterion benchmarks for the MM-PBSA energy components.
//!
//! Uses a synthetic topology and coordinates to benchmark each energy term
//! independently, as well as the full single-frame binding energy workflow.
//!
//! Run with: cargo bench -p rst-mmpbsa

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rst_core::amber::prmtop::AmberTopology;
use rst_mmpbsa::binding::{BindingConfig, TrajectoryFormat};
use rst_mmpbsa::gb_energy::{GbModel, GbParams};
use rst_mmpbsa::mm_energy::{compute_mm_energy, compute_mm_energy_with_nb, PairBitmap};
use rst_mmpbsa::sa_energy::{compute_sa_energy, SaParams};
use std::f64::consts::PI;
use std::sync::Arc;

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
    // Build a simple exclusion list where each atom excludes its bonded partners
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

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_mm_energy(c: &mut Criterion) {
    let mut group = c.benchmark_group("mm_energy");
    // Test multiple system sizes
    for &(n_res, atoms_per_res) in &[(10, 10), (30, 10), (100, 10)] {
        let top = build_synthetic_topology(n_res, atoms_per_res);
        let coords = build_synthetic_coords(top.n_atoms);
        let (excluded, pairs_14) = build_nb_sets(&top);

        let label = format!("{}atoms", top.n_atoms);

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

fn bench_gb_energy(c: &mut Criterion) {
    let mut group = c.benchmark_group("gb_energy");
    let gb_params = GbParams::default();

    for &(n_res, atoms_per_res) in &[(10, 10), (30, 10), (100, 10)] {
        let top = build_synthetic_topology(n_res, atoms_per_res);
        let coords = build_synthetic_coords(top.n_atoms);
        let label = format!("{}atoms", top.n_atoms);

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

    // Also benchmark OBC-II at the medium size
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

fn bench_sa_energy(c: &mut Criterion) {
    let mut group = c.benchmark_group("sa_energy");
    let sa_params = SaParams::default();

    for &(n_res, atoms_per_res) in &[(10, 10), (30, 10), (100, 10)] {
        let top = build_synthetic_topology(n_res, atoms_per_res);
        let coords = build_synthetic_coords(top.n_atoms);
        let label = format!("{}atoms", top.n_atoms);

        group.bench_with_input(
            BenchmarkId::new("shrake_rupley", &label),
            &(&top, &coords, &sa_params),
            |b, &(top, coords, params)| {
                b.iter(|| compute_sa_energy(black_box(top), black_box(coords), params));
            },
        );
    }
    group.finish();
}

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
            gb_params: GbParams::default(),
            sa_params: SaParams::default(),
            trajectory_format: TrajectoryFormat::Mdcrd { has_box: false },
            stride: 1,
            start_frame: 0,
            end_frame: usize::MAX,
        };

        group.bench_with_input(
            BenchmarkId::new("full_pipeline", &label),
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

/// Benchmark NB set construction separately, since it is a one-time cost
/// per topology that can dominate if done repeatedly.
fn bench_nb_set_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("nb_set_construction");

    for &(n_res, atoms_per_res) in &[(10, 10), (30, 10), (100, 10)] {
        let top = build_synthetic_topology(n_res, atoms_per_res);
        let label = format!("{}atoms", top.n_atoms);

        group.bench_with_input(
            BenchmarkId::new("exclusion_set", &label),
            &top,
            |b, top| {
                b.iter(|| rst_mmpbsa::mm_energy::build_exclusion_set(black_box(top)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("14_pairs", &label),
            &top,
            |b, top| {
                b.iter(|| rst_mmpbsa::mm_energy::build_14_pairs(black_box(top)));
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_mm_energy,
    bench_gb_energy,
    bench_sa_energy,
    bench_binding_single_frame,
    bench_nb_set_construction,
);
criterion_main!(benches);
