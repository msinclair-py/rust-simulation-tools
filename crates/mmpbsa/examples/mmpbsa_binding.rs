//! MM-GBSA binding free energy calculation example.
//!
//! Demonstrates the 1-trajectory approach for computing the binding free energy
//! of a protein-protein or protein-ligand complex using Generalized Born (GB)
//! solvation with per-residue decomposition and entropy estimation.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example mmpbsa_binding -- \
//!     --prmtop complex.prmtop \
//!     --mdcrd trajectory.mdcrd \
//!     --receptor-residues 0-165 \
//!     --ligand-residues 166-240
//! ```

use rst_mmpbsa::binding::{compute_binding_energy, BindingConfig, BindingResult, TrajectoryFormat};
use rst_mmpbsa::decomposition::decompose_binding_energy;
use rst_mmpbsa::entropy::interaction_entropy;
use rst_mmpbsa::gb_energy::{GbModel, GbParams};
use rst_mmpbsa::sa_energy::SaParams;
use rst_core::amber::prmtop::parse_prmtop;
use std::path::Path;

fn main() {
    // -------------------------------------------------------------------------
    // 1. Parse command-line arguments
    // -------------------------------------------------------------------------
    let args: Vec<String> = std::env::args().collect();
    let (prmtop_path, mdcrd_path, receptor_residues, ligand_residues) = parse_args(&args);

    println!("MM-GBSA Binding Free Energy Calculation");
    println!("========================================");
    println!("Topology: {}", prmtop_path);
    println!("Trajectory: {}", mdcrd_path);
    println!(
        "Receptor: residues {}-{} ({} residues)",
        receptor_residues.first().unwrap(),
        receptor_residues.last().unwrap(),
        receptor_residues.len()
    );
    println!(
        "Ligand: residues {}-{} ({} residues)",
        ligand_residues.first().unwrap(),
        ligand_residues.last().unwrap(),
        ligand_residues.len()
    );
    println!();

    // -------------------------------------------------------------------------
    // 2. Load the AMBER topology
    // -------------------------------------------------------------------------
    let topology = parse_prmtop(&prmtop_path).expect("Failed to parse prmtop");
    println!(
        "Loaded topology: {} atoms, {} residues",
        topology.n_atoms, topology.n_residues
    );

    // -------------------------------------------------------------------------
    // 3. Configure the MM-GBSA calculation
    // -------------------------------------------------------------------------
    let config = BindingConfig {
        receptor_residues: receptor_residues.clone(),
        ligand_residues: ligand_residues.clone(),
        // OBC-II (igb=5) is generally the most accurate GB model
        gb_params: GbParams {
            model: GbModel::ObcII,
            solute_dielectric: 1.0,
            solvent_dielectric: 80.0,
            salt_concentration: 0.15, // 150 mM NaCl
            ..GbParams::default()
        },
        // Standard LCPO surface tension parameters
        sa_params: SaParams {
            surface_tension: 0.0072, // kcal/(mol·Å²)
            offset: 0.0,
            probe_radius: 1.4,       // Å
            n_sphere_points: 960,
        },
        trajectory_format: TrajectoryFormat::Mdcrd { has_box: false },
        stride: 1,
        start_frame: 0,
        end_frame: usize::MAX,
    };

    // -------------------------------------------------------------------------
    // 4. Run the binding free energy calculation over the trajectory
    // -------------------------------------------------------------------------
    println!("\nRunning MM-GBSA calculation...");
    let result: BindingResult =
        compute_binding_energy(&topology, Path::new(&mdcrd_path), &config)
            .expect("Binding energy calculation failed");

    // -------------------------------------------------------------------------
    // 5. Print summary statistics
    // -------------------------------------------------------------------------
    println!("\n{}", "=".repeat(60));
    println!("MM-GBSA Binding Free Energy Results ({} frames)", result.frames.len());
    println!("{}", "=".repeat(60));
    println!(
        "{:<20} {:>12} kcal/mol",
        "ΔE(MM)", format!("{:.2}", result.mean_delta_mm)
    );
    println!(
        "{:<20} {:>12} kcal/mol",
        "ΔG(GB)", format!("{:.2}", result.mean_delta_gb)
    );
    println!(
        "{:<20} {:>12} kcal/mol",
        "ΔG(SA)", format!("{:.2}", result.mean_delta_sa)
    );
    println!("{}", "-".repeat(60));
    println!(
        "{:<20} {:>12} kcal/mol",
        "ΔG(total)", format!("{:.2}", result.mean_delta_total)
    );
    println!(
        "{:<20} {:>12} kcal/mol",
        "σ(ΔG)", format!("{:.2}", result.std_delta_total)
    );

    // -------------------------------------------------------------------------
    // 6. Estimate entropic contribution via Interaction Entropy
    // -------------------------------------------------------------------------
    if let Some(entropy) = interaction_entropy(&result.frames, 298.15) {
        println!();
        println!(
            "{:<20} {:>12} kcal/mol",
            "-TΔS(IE)", format!("{:.2}", entropy.minus_tds)
        );
        println!(
            "{:<20} {:>12} kcal/mol",
            "ΔG(bind) = ΔH - TΔS",
            format!("{:.2}", result.mean_delta_total + entropy.minus_tds)
        );
    }

    // -------------------------------------------------------------------------
    // 7. Per-residue decomposition on the last frame
    // -------------------------------------------------------------------------
    println!("\n{}", "=".repeat(60));
    println!("Per-Residue Decomposition (last frame)");
    println!("{}", "=".repeat(60));

    let last_frame_coords = &result.last_frame_coords;

    let decomp = decompose_binding_energy(
        &topology,
        &last_frame_coords,
        &receptor_residues,
        &ligand_residues,
        &config.gb_params,
        &config.sa_params,
    )
    .expect("Decomposition failed");

    // Print top contributing receptor residues (sorted by |total|)
    let mut receptor_contribs = decomp.receptor_residues.clone();
    receptor_contribs.sort_by(|a, b| {
        b.total().abs().partial_cmp(&a.total().abs()).unwrap()
    });

    println!(
        "\n{:<6} {:<8} {:>8} {:>8} {:>8} {:>8} {:>10}",
        "ResID", "Name", "vdW", "Elec", "GB", "SA", "Total"
    );
    println!("{}", "-".repeat(60));

    for contrib in receptor_contribs.iter().take(15) {
        println!(
            "{:<6} {:<8} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>10.2}",
            contrib.residue_index + 1,
            contrib.residue_label,
            contrib.vdw,
            contrib.elec,
            contrib.gb,
            contrib.sa,
            contrib.total(),
        );
    }

    // Print top contributing ligand residues
    let mut ligand_contribs = decomp.ligand_residues.clone();
    ligand_contribs.sort_by(|a, b| {
        b.total().abs().partial_cmp(&a.total().abs()).unwrap()
    });

    println!("\nTop ligand residues:");
    println!(
        "{:<6} {:<8} {:>8} {:>8} {:>8} {:>8} {:>10}",
        "ResID", "Name", "vdW", "Elec", "GB", "SA", "Total"
    );
    println!("{}", "-".repeat(60));

    for contrib in ligand_contribs.iter().take(15) {
        println!(
            "{:<6} {:<8} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>10.2}",
            contrib.residue_index + 1,
            contrib.residue_label,
            contrib.vdw,
            contrib.elec,
            contrib.gb,
            contrib.sa,
            contrib.total(),
        );
    }

    // -------------------------------------------------------------------------
    // 8. Per-frame energy time series
    // -------------------------------------------------------------------------
    println!("\n{}", "=".repeat(60));
    println!("Per-Frame Energy Time Series");
    println!("{}", "=".repeat(60));
    println!(
        "{:<8} {:>10} {:>10} {:>10} {:>10}",
        "Frame", "ΔMM", "ΔGB", "ΔSA", "ΔTotal"
    );
    println!("{}", "-".repeat(60));

    for (i, frame) in result.frames.iter().enumerate() {
        println!(
            "{:<8} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
            i + 1,
            frame.delta_mm,
            frame.delta_gb,
            frame.delta_sa,
            frame.delta_total,
        );
    }
}

/// Parse command-line arguments into (prmtop, mdcrd, receptor_residues, ligand_residues).
fn parse_args(args: &[String]) -> (String, String, Vec<usize>, Vec<usize>) {
    let mut prmtop = None;
    let mut mdcrd = None;
    let mut receptor = None;
    let mut ligand = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--prmtop" => {
                prmtop = Some(args[i + 1].clone());
                i += 2;
            }
            "--mdcrd" => {
                mdcrd = Some(args[i + 1].clone());
                i += 2;
            }
            "--receptor-residues" => {
                receptor = Some(parse_residue_range(&args[i + 1]));
                i += 2;
            }
            "--ligand-residues" => {
                ligand = Some(parse_residue_range(&args[i + 1]));
                i += 2;
            }
            "--help" | "-h" => {
                eprintln!(
                    "Usage: mmpbsa_binding --prmtop <FILE> --mdcrd <FILE> \
                     --receptor-residues <START-END> --ligand-residues <START-END>"
                );
                eprintln!();
                eprintln!("Residue ranges are 0-based and inclusive (e.g., 0-165).");
                std::process::exit(0);
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
    }

    (
        prmtop.expect("--prmtop is required"),
        mdcrd.expect("--mdcrd is required"),
        receptor.expect("--receptor-residues is required"),
        ligand.expect("--ligand-residues is required"),
    )
}

/// Parse "START-END" into a Vec of 0-based residue indices.
fn parse_residue_range(s: &str) -> Vec<usize> {
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 2 {
        eprintln!("Expected residue range as START-END (e.g., 0-165), got: {}", s);
        std::process::exit(1);
    }
    let start: usize = parts[0].parse().expect("Invalid start residue");
    let end: usize = parts[1].parse().expect("Invalid end residue");
    (start..=end).collect()
}
