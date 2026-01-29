//! Lexer for the atom selection language.

use crate::selection::error::SelectionError;

/// A token with its byte span in the input string.
#[derive(Debug, Clone)]
pub struct SpannedToken {
    pub token: Token,
    pub span: (usize, usize),
}

/// Token types for the selection language.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // String keywords
    Name,
    Resname,
    Type,
    // Range keywords
    Resid,
    Index,
    // Numeric keywords
    Mass,
    Charge,
    Radius,
    Sigma,
    Epsilon,
    // Boolean operators
    And,
    Or,
    Not,
    Within,
    Of,
    // Convenience keywords
    Protein,
    Water,
    Backbone,
    Sidechain,
    Hydrogen,
    All,
    None_,
    // Literals
    Ident(String),
    Integer(i64),
    Float(f64),
    // Punctuation / operators
    LParen,
    RParen,
    Gt,
    Lt,
    Ge,
    Le,
    Eq,
    Ne,
    Comma,
    Dash,
    Colon,
    // End
    Eof,
}

/// Lexer that tokenizes a selection expression string.
pub struct Lexer<'a> {
    input: &'a str,
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            input,
            bytes: input.as_bytes(),
            pos: 0,
        }
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn err(&self, msg: impl Into<String>, start: usize) -> SelectionError {
        SelectionError::new(msg)
            .with_span(start, self.pos)
            .with_input(self.input.to_string())
    }

    pub fn tokenize(&mut self) -> Result<Vec<SpannedToken>, SelectionError> {
        let mut tokens = Vec::new();
        loop {
            self.skip_whitespace();
            let start = self.pos;
            if self.pos >= self.bytes.len() {
                tokens.push(SpannedToken {
                    token: Token::Eof,
                    span: (start, start),
                });
                break;
            }
            let ch = self.bytes[self.pos];
            let token = match ch {
                b'(' => {
                    self.pos += 1;
                    Token::LParen
                }
                b')' => {
                    self.pos += 1;
                    Token::RParen
                }
                b',' => {
                    self.pos += 1;
                    Token::Comma
                }
                b':' => {
                    self.pos += 1;
                    Token::Colon
                }
                b'>' => {
                    self.pos += 1;
                    if self.pos < self.bytes.len() && self.bytes[self.pos] == b'=' {
                        self.pos += 1;
                        Token::Ge
                    } else {
                        Token::Gt
                    }
                }
                b'<' => {
                    self.pos += 1;
                    if self.pos < self.bytes.len() && self.bytes[self.pos] == b'=' {
                        self.pos += 1;
                        Token::Le
                    } else {
                        Token::Lt
                    }
                }
                b'=' => {
                    self.pos += 1;
                    if self.pos < self.bytes.len() && self.bytes[self.pos] == b'=' {
                        self.pos += 1;
                        Token::Eq
                    } else {
                        return Err(self.err("Expected '==' operator", start));
                    }
                }
                b'!' => {
                    self.pos += 1;
                    if self.pos < self.bytes.len() && self.bytes[self.pos] == b'=' {
                        self.pos += 1;
                        Token::Ne
                    } else {
                        return Err(self.err("Expected '!=' operator", start));
                    }
                }
                b'-' => {
                    self.pos += 1;
                    Token::Dash
                }
                b'0'..=b'9' => self.lex_number()?,
                b'a'..=b'z' | b'A'..=b'Z' | b'_' | b'*' | b'?' => self.lex_word()?,
                _ => return Err(self.err(format!("Unexpected character '{}'", ch as char), start)),
            };
            tokens.push(SpannedToken {
                token,
                span: (start, self.pos),
            });
        }
        Ok(tokens)
    }

    fn lex_number(&mut self) -> Result<Token, SelectionError> {
        let start = self.pos;
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        if self.pos < self.bytes.len() && self.bytes[self.pos] == b'.' {
            self.pos += 1;
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
            let s = &self.input[start..self.pos];
            let val: f64 = s
                .parse()
                .map_err(|_| self.err(format!("Invalid float '{}'", s), start))?;
            Ok(Token::Float(val))
        } else {
            let s = &self.input[start..self.pos];
            let val: i64 = s
                .parse()
                .map_err(|_| self.err(format!("Invalid integer '{}'", s), start))?;
            Ok(Token::Integer(val))
        }
    }

    fn lex_word(&mut self) -> Result<Token, SelectionError> {
        let start = self.pos;
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            if b.is_ascii_alphanumeric() || b == b'_' || b == b'*' || b == b'?' || b == b'\'' {
                self.pos += 1;
            } else {
                break;
            }
        }
        let word = &self.input[start..self.pos];
        let token = match word.to_lowercase().as_str() {
            "name" => Token::Name,
            "resname" => Token::Resname,
            "type" => Token::Type,
            "resid" => Token::Resid,
            "index" => Token::Index,
            "mass" => Token::Mass,
            "charge" => Token::Charge,
            "radius" => Token::Radius,
            "sigma" => Token::Sigma,
            "epsilon" => Token::Epsilon,
            "and" => Token::And,
            "or" => Token::Or,
            "not" => Token::Not,
            "within" => Token::Within,
            "of" => Token::Of,
            "protein" => Token::Protein,
            "water" => Token::Water,
            "backbone" => Token::Backbone,
            "sidechain" => Token::Sidechain,
            "hydrogen" => Token::Hydrogen,
            "all" => Token::All,
            "none" => Token::None_,
            _ => Token::Ident(word.to_string()),
        };
        Ok(token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokenize() {
        let mut lexer = Lexer::new("name CA");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens.len(), 3); // Name, Ident("CA"), Eof
        assert_eq!(tokens[0].token, Token::Name);
        assert!(matches!(&tokens[1].token, Token::Ident(s) if s == "CA"));
        assert_eq!(tokens[2].token, Token::Eof);
    }

    #[test]
    fn test_numeric_tokens() {
        let mut lexer = Lexer::new("mass > 12.0");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[0].token, Token::Mass);
        assert_eq!(tokens[1].token, Token::Gt);
        assert_eq!(tokens[2].token, Token::Float(12.0));
    }

    #[test]
    fn test_range_tokens() {
        let mut lexer = Lexer::new("resid 1-10");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[0].token, Token::Resid);
        assert_eq!(tokens[1].token, Token::Integer(1));
        assert_eq!(tokens[2].token, Token::Dash);
        assert_eq!(tokens[3].token, Token::Integer(10));
    }

    #[test]
    fn test_comparison_ops() {
        let mut lexer = Lexer::new(">= <= == !=");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens[0].token, Token::Ge);
        assert_eq!(tokens[1].token, Token::Le);
        assert_eq!(tokens[2].token, Token::Eq);
        assert_eq!(tokens[3].token, Token::Ne);
    }
}
