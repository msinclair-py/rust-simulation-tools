// Standalone program to run our antechamber implementation on test molecules.
// Build: cargo build --bin validation_runner
// (Or just run via cargo test)

use std::path::Path;

fn main() {
    let test_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("antechamber_validation");

    let molecules = ["methanol", "acetic_acid", "benzene", "methylamine"];

    for mol_name in &molecules {
        println!("============================================");
        println!("Processing: {}", mol_name);
        println!("============================================");

        let sdf_path = test_dir.join(format!("{}.sdf", mol_name));
        let content = std::fs::read_to_string(&sdf_path).expect("Failed to read SDF");
        let mols = rst_core::sdf::parse_sdf(&content).expect("Failed to parse SDF");
        let sdf_mol = &mols[0];

        let ac_mol = rst_antechamber::molecule::AcMolecule::from_sdf(sdf_mol);

        let config = rst_antechamber::AntechamberConfig {
            net_charge: 0,
            charge_method: rst_antechamber::ChargeMethod::Am1Bcc,
        };

        match rst_antechamber::run_antechamber(ac_mol, &config) {
            Ok(result) => {
                println!("\n--- Atom types and charges (RST) ---");
                for (i, atom) in result.molecule.atoms.iter().enumerate() {
                    println!(
                        "  {:>3}  {:<4}  type={:<6}  charge={:>10.6}",
                        i + 1,
                        atom.element,
                        atom.gaff2_type,
                        atom.charge
                    );
                }
            }
            Err(e) => {
                println!("FAILED: {}", e);
            }
        }
        println!();
    }
}
