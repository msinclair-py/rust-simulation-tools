//! Selection language error types with position spans.

use std::fmt;

/// Error type for the atom selection language.
#[derive(Debug, Clone)]
pub struct SelectionError {
    pub message: String,
    pub span: Option<(usize, usize)>,
    pub input: Option<String>,
}

impl SelectionError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            span: None,
            input: None,
        }
    }

    pub fn with_span(mut self, start: usize, end: usize) -> Self {
        self.span = Some((start, end));
        self
    }

    pub fn with_input(mut self, input: impl Into<String>) -> Self {
        self.input = Some(input.into());
        self
    }
}

impl fmt::Display for SelectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SelectionError: {}", self.message)?;
        if let (Some((start, end)), Some(input)) = (self.span, &self.input) {
            write!(f, "\n  {}", input)?;
            write!(
                f,
                "\n  {}{}",
                " ".repeat(start),
                "^".repeat(end.saturating_sub(start).max(1))
            )?;
        }
        Ok(())
    }
}

impl std::error::Error for SelectionError {}
