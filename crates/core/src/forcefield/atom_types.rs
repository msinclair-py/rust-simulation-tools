//! Atom type utilities.
//!
//! Provides element lookup by AMBER atom type name. The mapping covers
//! standard protein types (ff14SB/ff19SB), GAFF2 small-molecule types,
//! and Lipid21 types.

/// Map an AMBER atom type name to its element symbol.
///
/// Returns `None` for virtual sites (extra points / lone pairs such as `"EP"`
/// and `"LP"`) and for unrecognized atom types.
#[allow(clippy::too_many_lines)]
pub fn element_from_type(atom_type: &str) -> Option<&'static str> {
    match atom_type {
        // ---- Hydrogen types ----
        // Standard AMBER protein hydrogen types
        "H" | "HO" | "HS" | "H1" | "H2" | "H3" | "H4" | "H5" | "HW" | "HC" | "HA" | "HP"
        | "HZ" => Some("H"),
        // GAFF2 lowercase hydrogen types
        "h1" | "h2" | "h3" | "h4" | "h5" | "ha" | "hc" | "hn" | "ho" | "hp" | "hs" | "hw"
        | "hx" => Some("H"),
        // Lipid21 hydrogen types
        "hA" | "hB" | "hE" | "hL" | "hN" | "hO" | "hX" => Some("H"),

        // ---- Carbon types ----
        // Standard AMBER protein carbon types
        "C" | "CA" | "CB" | "CC" | "CD" | "CK" | "CM" | "CN" | "CQ" | "CR" | "CT" | "CV"
        | "CW" | "C*" | "CX" | "XC" | "CY" | "CZ" | "CS" | "CP" | "CI" | "CJ" | "C5"
        | "C4" | "CH" | "CO" | "C8" | "2C" | "3C" => Some("C"),
        // GAFF2 lowercase carbon types
        "c" | "c1" | "c2" | "c3" | "c5" | "c6" | "ca" | "cc" | "cd" | "ce" | "cf" | "cg"
        | "ch" | "cp" | "cs" | "cq" | "cu" | "cv" | "cx" | "cy" | "cz" => Some("C"),
        // Lipid21 carbon types
        "cA" | "cB" | "cC" | "cD" => Some("C"),

        // ---- Nitrogen types ----
        // Standard AMBER protein nitrogen types
        "N" | "NA" | "NB" | "NC" | "N2" | "N*" | "N3" | "NT" | "NY" | "NP" | "NQ" => {
            Some("N")
        }
        // GAFF2 lowercase nitrogen types
        "n" | "n1" | "n2" | "n3" | "n4" | "n5" | "n6" | "n7" | "n8" | "n9" | "na" | "nb"
        | "nc" | "nd" | "ne" | "nf" | "nh" | "ni" | "nj" | "nk" | "nl" | "nm" | "nn"
        | "no" | "np" | "nq" | "ns" | "nt" | "nu" | "nv" | "nx" | "ny" | "nz" | "n+" => {
            Some("N")
        }
        // Lipid21 nitrogen types
        "nA" | "nN" => Some("N"),

        // ---- Oxygen types ----
        // Standard AMBER protein oxygen types
        "O" | "O2" | "OH" | "OS" | "OP" | "OW" => Some("O"),
        // GAFF2 lowercase oxygen types
        "o" | "o2" | "oh" | "op" | "oq" | "os" | "ow" => Some("O"),
        // Lipid21 oxygen types
        "oC" | "oH" | "oO" | "oP" | "oS" | "oT" => Some("O"),

        // ---- Sulfur types ----
        "S" | "SH" => Some("S"),
        "s" | "s2" | "s3" | "s4" | "s6" | "sh" | "sp" | "sq" | "ss" | "sx" | "sy" => {
            Some("S")
        }

        // ---- Phosphorus types ----
        "P" => Some("P"),
        "p2" | "p3" | "p4" | "p5" | "pb" | "pc" | "pd" | "pe" | "pf" | "px" | "py" => {
            Some("P")
        }
        "pA" => Some("P"),

        // ---- Halogens ----
        "F" | "f" | "F-" => Some("F"),
        "Cl" | "cl" | "Cl-" => Some("Cl"),
        "Br" | "br" | "Br-" => Some("Br"),
        "I" | "i" | "I-" => Some("I"),

        // ---- Metals / Ions ----
        "MG" | "Mg+" | "Mg2+" => Some("Mg"),
        "C0" | "Ca2+" => Some("Ca"),
        "Zn" | "Zn2+" => Some("Zn"),
        "Fe2+" | "Fe3+" => Some("Fe"),
        "Na+" => Some("Na"),
        "K+" => Some("K"),
        "Li+" => Some("Li"),
        "Rb+" => Some("Rb"),
        "Cs+" => Some("Cs"),

        // ---- Extra points / lone pairs (no element) ----
        "EP" | "LP" => None,

        // Unknown type
        _ => None,
    }
}
