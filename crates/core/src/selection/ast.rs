//! AST node types for the atom selection language.

/// Top-level expression node.
#[derive(Debug, Clone)]
pub enum Expr {
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Not(Box<Expr>),
    NameMatch { field: StringField, pattern: StringPattern },
    NumericCmp { field: NumericField, op: CmpOp, value: f64 },
    RangeSelect { field: RangeField, ranges: Vec<RangeSpec> },
    Within { distance: f64, inner: Box<Expr> },
    Convenience(ConvenienceKeyword),
}

/// Fields that match against string values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringField {
    Name,
    Resname,
    Type,
}

/// String matching pattern (supports * and ? globs).
#[derive(Debug, Clone)]
pub enum StringPattern {
    Exact(String),
    Glob(String),
}

impl StringPattern {
    pub fn matches(&self, s: &str) -> bool {
        match self {
            StringPattern::Exact(pat) => s == pat,
            StringPattern::Glob(pat) => glob_match(pat, s),
        }
    }
}

/// Simple glob matcher supporting * and ?.
fn glob_match(pattern: &str, text: &str) -> bool {
    let pat: Vec<char> = pattern.chars().collect();
    let txt: Vec<char> = text.chars().collect();
    let (mut pi, mut ti) = (0, 0);
    let (mut star_pi, mut star_ti) = (usize::MAX, 0);

    while ti < txt.len() {
        if pi < pat.len() && (pat[pi] == '?' || pat[pi] == txt[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < pat.len() && pat[pi] == '*' {
            star_pi = pi;
            star_ti = ti;
            pi += 1;
        } else if star_pi != usize::MAX {
            pi = star_pi + 1;
            star_ti += 1;
            ti = star_ti;
        } else {
            return false;
        }
    }
    while pi < pat.len() && pat[pi] == '*' {
        pi += 1;
    }
    pi == pat.len()
}

/// Fields that use numeric comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericField {
    Mass,
    Charge,
    Radius,
    Sigma,
    Epsilon,
}

/// Comparison operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Gt,
    Lt,
    Ge,
    Le,
    Eq,
    Ne,
}

impl CmpOp {
    pub fn compare(&self, lhs: f64, rhs: f64) -> bool {
        match self {
            CmpOp::Gt => lhs > rhs,
            CmpOp::Lt => lhs < rhs,
            CmpOp::Ge => lhs >= rhs,
            CmpOp::Le => lhs <= rhs,
            CmpOp::Eq => (lhs - rhs).abs() < 1e-9,
            CmpOp::Ne => (lhs - rhs).abs() >= 1e-9,
        }
    }
}

/// Fields for range-based integer selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeField {
    /// 1-based residue ID (AMBER/PDB convention)
    Resid,
    /// 0-based atom index
    Index,
}

/// A single range specification.
#[derive(Debug, Clone)]
pub enum RangeSpec {
    Single(i64),
    Range(i64, i64),
}

impl RangeSpec {
    pub fn contains(&self, value: i64) -> bool {
        match self {
            RangeSpec::Single(v) => *v == value,
            RangeSpec::Range(lo, hi) => value >= *lo && value <= *hi,
        }
    }
}

/// Convenience keyword selectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvenienceKeyword {
    Protein,
    Water,
    Backbone,
    Sidechain,
    Hydrogen,
    All,
    None,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glob_exact() {
        assert!(glob_match("CA", "CA"));
        assert!(!glob_match("CA", "CB"));
    }

    #[test]
    fn test_glob_star() {
        assert!(glob_match("C*", "CA"));
        assert!(glob_match("C*", "CB"));
        assert!(glob_match("C*", "C"));
        assert!(!glob_match("C*", "NA"));
    }

    #[test]
    fn test_glob_question() {
        assert!(glob_match("C?", "CA"));
        assert!(!glob_match("C?", "C"));
        assert!(!glob_match("C?", "CAB"));
    }

    #[test]
    fn test_range_spec() {
        assert!(RangeSpec::Single(5).contains(5));
        assert!(!RangeSpec::Single(5).contains(6));
        assert!(RangeSpec::Range(1, 10).contains(5));
        assert!(!RangeSpec::Range(1, 10).contains(11));
    }
}
