//! Recursive descent parser for the atom selection language.

use crate::selection::ast::*;
use crate::selection::error::SelectionError;
use crate::selection::token::*;

/// Parser state wrapping a token stream.
pub struct Parser {
    tokens: Vec<SpannedToken>,
    pos: usize,
    input: String,
}

impl Parser {
    pub fn new(tokens: Vec<SpannedToken>, input: String) -> Self {
        Self {
            tokens,
            pos: 0,
            input,
        }
    }

    pub fn parse(mut self) -> Result<Expr, SelectionError> {
        let expr = self.parse_or()?;
        if !self.at_eof() {
            let span = self.current_span();
            return Err(SelectionError::new(format!(
                "Unexpected token {:?}",
                self.current().token
            ))
            .with_span(span.0, span.1)
            .with_input(self.input.clone()));
        }
        Ok(expr)
    }

    fn current(&self) -> &SpannedToken {
        &self.tokens[self.pos.min(self.tokens.len() - 1)]
    }

    fn current_span(&self) -> (usize, usize) {
        self.current().span
    }

    fn at_eof(&self) -> bool {
        self.current().token == Token::Eof
    }

    fn advance(&mut self) {
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
    }

    fn expect(&mut self, expected: &Token) -> Result<SpannedToken, SelectionError> {
        let tok = self.current().clone();
        if std::mem::discriminant(&tok.token) == std::mem::discriminant(expected) {
            self.advance();
            Ok(tok)
        } else {
            Err(
                SelectionError::new(format!("Expected {:?}, found {:?}", expected, tok.token))
                    .with_span(tok.span.0, tok.span.1)
                    .with_input(self.input.clone()),
            )
        }
    }

    // or_expr = and_expr ("or" and_expr)*
    fn parse_or(&mut self) -> Result<Expr, SelectionError> {
        let mut left = self.parse_and()?;
        while self.current().token == Token::Or {
            self.advance();
            let right = self.parse_and()?;
            left = Expr::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    // and_expr = not_expr ("and" not_expr)*
    fn parse_and(&mut self) -> Result<Expr, SelectionError> {
        let mut left = self.parse_not()?;
        while self.current().token == Token::And {
            self.advance();
            let right = self.parse_not()?;
            left = Expr::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    // not_expr = "not" not_expr | atom_expr
    fn parse_not(&mut self) -> Result<Expr, SelectionError> {
        if self.current().token == Token::Not {
            self.advance();
            let inner = self.parse_not()?;
            Ok(Expr::Not(Box::new(inner)))
        } else {
            self.parse_atom()
        }
    }

    // atom_expr = "(" selection ")" | within_expr | keyword_expr | convenience
    fn parse_atom(&mut self) -> Result<Expr, SelectionError> {
        match &self.current().token {
            Token::LParen => {
                self.advance();
                let inner = self.parse_or()?;
                self.expect(&Token::RParen)?;
                Ok(inner)
            }
            Token::Within => self.parse_within(),
            Token::Name | Token::Resname | Token::Type => self.parse_string_kw(),
            Token::Mass | Token::Charge | Token::Radius | Token::Sigma | Token::Epsilon => {
                self.parse_numeric_kw()
            }
            Token::Resid | Token::Index => self.parse_range_kw(),
            Token::Protein => {
                self.advance();
                Ok(Expr::Convenience(ConvenienceKeyword::Protein))
            }
            Token::Water => {
                self.advance();
                Ok(Expr::Convenience(ConvenienceKeyword::Water))
            }
            Token::Backbone => {
                self.advance();
                Ok(Expr::Convenience(ConvenienceKeyword::Backbone))
            }
            Token::Sidechain => {
                self.advance();
                Ok(Expr::Convenience(ConvenienceKeyword::Sidechain))
            }
            Token::Hydrogen => {
                self.advance();
                Ok(Expr::Convenience(ConvenienceKeyword::Hydrogen))
            }
            Token::All => {
                self.advance();
                Ok(Expr::Convenience(ConvenienceKeyword::All))
            }
            Token::None_ => {
                self.advance();
                Ok(Expr::Convenience(ConvenienceKeyword::None))
            }
            _ => {
                let span = self.current_span();
                Err(SelectionError::new(format!(
                    "Expected selection expression, found {:?}",
                    self.current().token
                ))
                .with_span(span.0, span.1)
                .with_input(self.input.clone()))
            }
        }
    }

    // within FLOAT of atom_expr
    fn parse_within(&mut self) -> Result<Expr, SelectionError> {
        self.advance(); // consume 'within'
        let distance = match &self.current().token {
            Token::Float(f) => {
                let v = *f;
                self.advance();
                v
            }
            Token::Integer(i) => {
                let v = *i as f64;
                self.advance();
                v
            }
            _ => {
                let span = self.current_span();
                return Err(SelectionError::new("Expected distance after 'within'")
                    .with_span(span.0, span.1)
                    .with_input(self.input.clone()));
            }
        };
        self.expect(&Token::Of)?;
        let inner = self.parse_atom()?;
        Ok(Expr::Within {
            distance,
            inner: Box::new(inner),
        })
    }

    // string_kw string_arg
    fn parse_string_kw(&mut self) -> Result<Expr, SelectionError> {
        let field = match self.current().token {
            Token::Name => StringField::Name,
            Token::Resname => StringField::Resname,
            Token::Type => StringField::Type,
            _ => unreachable!(),
        };
        self.advance();
        let pattern = self.parse_string_arg()?;
        Ok(Expr::NameMatch { field, pattern })
    }

    fn parse_string_arg(&mut self) -> Result<StringPattern, SelectionError> {
        match &self.current().token {
            Token::Ident(s) => {
                let s = s.clone();
                self.advance();
                if s.contains('*') || s.contains('?') {
                    Ok(StringPattern::Glob(s))
                } else {
                    Ok(StringPattern::Exact(s))
                }
            }
            _ => {
                let span = self.current_span();
                Err(SelectionError::new("Expected identifier after keyword")
                    .with_span(span.0, span.1)
                    .with_input(self.input.clone()))
            }
        }
    }

    // numeric_kw cmp_op NUMBER
    fn parse_numeric_kw(&mut self) -> Result<Expr, SelectionError> {
        let field = match self.current().token {
            Token::Mass => NumericField::Mass,
            Token::Charge => NumericField::Charge,
            Token::Radius => NumericField::Radius,
            Token::Sigma => NumericField::Sigma,
            Token::Epsilon => NumericField::Epsilon,
            _ => unreachable!(),
        };
        self.advance();
        let op = self.parse_cmp_op()?;

        // Handle negative numbers: if we see Dash followed by a number
        let negative = if self.current().token == Token::Dash {
            self.advance();
            true
        } else {
            false
        };

        let value = match &self.current().token {
            Token::Float(f) => {
                let v = *f;
                self.advance();
                v
            }
            Token::Integer(i) => {
                let v = *i as f64;
                self.advance();
                v
            }
            _ => {
                let span = self.current_span();
                return Err(
                    SelectionError::new("Expected number after comparison operator")
                        .with_span(span.0, span.1)
                        .with_input(self.input.clone()),
                );
            }
        };
        let value = if negative { -value } else { value };
        Ok(Expr::NumericCmp { field, op, value })
    }

    fn parse_cmp_op(&mut self) -> Result<CmpOp, SelectionError> {
        let op = match self.current().token {
            Token::Gt => CmpOp::Gt,
            Token::Lt => CmpOp::Lt,
            Token::Ge => CmpOp::Ge,
            Token::Le => CmpOp::Le,
            Token::Eq => CmpOp::Eq,
            Token::Ne => CmpOp::Ne,
            _ => {
                let span = self.current_span();
                return Err(SelectionError::new(
                    "Expected comparison operator (>, <, >=, <=, ==, !=)",
                )
                .with_span(span.0, span.1)
                .with_input(self.input.clone()));
            }
        };
        self.advance();
        Ok(op)
    }

    // range_kw range_arg
    fn parse_range_kw(&mut self) -> Result<Expr, SelectionError> {
        let field = match self.current().token {
            Token::Resid => RangeField::Resid,
            Token::Index => RangeField::Index,
            _ => unreachable!(),
        };
        self.advance();
        let ranges = self.parse_range_arg()?;
        Ok(Expr::RangeSelect { field, ranges })
    }

    // range_arg = INT "-" INT | INT ":" INT | INT ("," INT)* | INT
    fn parse_range_arg(&mut self) -> Result<Vec<RangeSpec>, SelectionError> {
        let mut specs = Vec::new();
        let first = self.expect_int()?;

        // Check for range delimiter
        if self.current().token == Token::Dash || self.current().token == Token::Colon {
            self.advance();
            let second = self.expect_int()?;
            specs.push(RangeSpec::Range(first, second));
        } else if self.current().token == Token::Comma {
            specs.push(RangeSpec::Single(first));
            while self.current().token == Token::Comma {
                self.advance();
                let val = self.expect_int()?;
                specs.push(RangeSpec::Single(val));
            }
        } else {
            specs.push(RangeSpec::Single(first));
        }
        Ok(specs)
    }

    fn expect_int(&mut self) -> Result<i64, SelectionError> {
        match &self.current().token {
            Token::Integer(i) => {
                let v = *i;
                self.advance();
                Ok(v)
            }
            _ => {
                let span = self.current_span();
                Err(SelectionError::new("Expected integer")
                    .with_span(span.0, span.1)
                    .with_input(self.input.clone()))
            }
        }
    }
}

/// Parse a selection expression string into an AST.
pub fn parse_selection(input: &str) -> Result<Expr, SelectionError> {
    let mut lexer = Lexer::new(input);
    let tokens = lexer.tokenize()?;
    let parser = Parser::new(tokens, input.to_string());
    parser.parse()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_name() {
        let expr = parse_selection("name CA").unwrap();
        assert!(matches!(
            expr,
            Expr::NameMatch {
                field: StringField::Name,
                ..
            }
        ));
    }

    #[test]
    fn test_parse_and_or() {
        let expr = parse_selection("name CA and resname ALA").unwrap();
        assert!(matches!(expr, Expr::And(_, _)));

        let expr = parse_selection("name CA or name CB").unwrap();
        assert!(matches!(expr, Expr::Or(_, _)));
    }

    #[test]
    fn test_parse_not() {
        let expr = parse_selection("not protein").unwrap();
        assert!(matches!(expr, Expr::Not(_)));
    }

    #[test]
    fn test_parse_parens() {
        let expr = parse_selection("(name CA or name CB) and protein").unwrap();
        assert!(matches!(expr, Expr::And(_, _)));
    }

    #[test]
    fn test_parse_resid_range() {
        let expr = parse_selection("resid 1-10").unwrap();
        assert!(matches!(
            expr,
            Expr::RangeSelect {
                field: RangeField::Resid,
                ..
            }
        ));
    }

    #[test]
    fn test_parse_numeric() {
        let expr = parse_selection("mass > 12.0").unwrap();
        assert!(matches!(
            expr,
            Expr::NumericCmp {
                field: NumericField::Mass,
                op: CmpOp::Gt,
                ..
            }
        ));
    }

    #[test]
    fn test_parse_negative_numeric() {
        let expr = parse_selection("charge < -0.5").unwrap();
        if let Expr::NumericCmp { value, .. } = expr {
            assert!((value - (-0.5)).abs() < 1e-9);
        } else {
            panic!("Expected NumericCmp");
        }
    }

    #[test]
    fn test_parse_within() {
        let expr = parse_selection("within 5.0 of resname LIG").unwrap();
        assert!(matches!(expr, Expr::Within { .. }));
    }

    #[test]
    fn test_parse_convenience() {
        assert!(matches!(
            parse_selection("protein").unwrap(),
            Expr::Convenience(ConvenienceKeyword::Protein)
        ));
        assert!(matches!(
            parse_selection("water").unwrap(),
            Expr::Convenience(ConvenienceKeyword::Water)
        ));
        assert!(matches!(
            parse_selection("all").unwrap(),
            Expr::Convenience(ConvenienceKeyword::All)
        ));
    }

    #[test]
    fn test_parse_complex() {
        let expr = parse_selection("protein and not backbone and resid 1-50").unwrap();
        assert!(matches!(expr, Expr::And(_, _)));
    }

    #[test]
    fn test_parse_resid_comma_list() {
        let expr = parse_selection("resid 1,3,5").unwrap();
        if let Expr::RangeSelect { ranges, .. } = expr {
            assert_eq!(ranges.len(), 3);
        } else {
            panic!("Expected RangeSelect");
        }
    }
}
