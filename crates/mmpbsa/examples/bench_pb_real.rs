//! Quick benchmark of PB binding energy on the real mmpbsa system.
//! Run: cargo run --release --example bench_pb_real

use rst_core::amber::inpcrd::parse_inpcrd;
use rst_core::amber::prmtop::parse_prmtop;
use rst_mmpbsa::binding::{
    compute_binding_energy_single_frame, BindingConfig, SolvationMethod, TrajectoryFormat,
};
use rst_mmpbsa::pb_energy::{compute_pb_energy_solvated, PbParams};
use rst_mmpbsa::sa_energy::SaParams;
use std::time::Instant;

fn main() {
    let topo = parse_prmtop("data/mmpbsa.prmtop").expect("Failed to parse prmtop");
    let inpcrd = parse_inpcrd("data/mmpbsa.inpcrd").expect("Failed to parse inpcrd");

    // Convert nm -> Angstroms
    let coords: Vec<[f64; 3]> = inpcrd
        .positions
        .iter()
        .map(|c| [c[0] * 10.0, c[1] * 10.0, c[2] * 10.0])
        .collect();

    println!(
        "System: {} atoms, {} residues",
        topo.n_atoms, topo.n_residues
    );

    let receptor_residues: Vec<usize> = (0..307).collect();
    let ligand_residues: Vec<usize> = (307..366).collect();
    let solute_residues: Vec<usize> = (0..366).collect();

    let pb_params = PbParams {
        grid_spacing: 0.5,
        grid_buffer: 10.0,
        solute_dielectric: 1.0,
        solvent_dielectric: 80.0,
        salt_concentration: 0.15,
        tolerance: 1e-6,
        max_iterations: 10000,
        ..PbParams::default()
    };

    // --- Standalone PB energy on the complex (solvated input) ---
    println!(
        "\n--- Standalone PB energy (complex, {} solute residues) ---",
        solute_residues.len()
    );
    let start = Instant::now();
    let pb = compute_pb_energy_solvated(&topo, &coords, &pb_params, Some(&solute_residues))
        .expect("PB energy failed");
    let elapsed = start.elapsed();
    println!("  PB energy: {:.2} kcal/mol", pb.total);
    println!("  Time: {:.2}s", elapsed.as_secs_f64());

    // --- Full MM-PBSA binding energy ---
    println!("\n--- Full MM-PBSA binding energy ---");
    let config = BindingConfig {
        receptor_residues,
        ligand_residues,
        solvation_method: SolvationMethod::PB(pb_params),
        sa_params: SaParams::default(),
        trajectory_format: TrajectoryFormat::Dcd,
        stride: 1,
        start_frame: 0,
        end_frame: usize::MAX,
    };

    let start = Instant::now();
    let result = compute_binding_energy_single_frame(&topo, &coords, &config)
        .expect("Binding energy failed");
    let elapsed = start.elapsed();

    println!("  Delta MM:    {:10.2} kcal/mol", result.delta_mm);
    println!("  Delta Polar: {:10.2} kcal/mol", result.delta_polar);
    println!("  Delta SA:    {:10.2} kcal/mol", result.delta_sa);
    println!("  Delta Total: {:10.2} kcal/mol", result.delta_total);
    println!("  Time: {:.2}s", elapsed.as_secs_f64());
}
