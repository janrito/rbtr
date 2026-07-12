//! Greeter — format greetings for named recipients.
//!
//! The Rust plugin extracts functions, structs/enums/unions/type
//! aliases/traits/modules/impl blocks (all as classes), methods inside
//! impl and trait blocks, `macro_rules!` (as functions), const/static (as
//! variables), and `use` imports. A struct and its impl both yield a
//! class chunk.

use std::collections::HashMap;

use crate::config::LOCALE;
use super::util;

/// The default greeting prefix.
const DEFAULT_GREETING: &str = "Hello";

static FALLBACK_LOCALE: &str = "en";

/// A greeter holding a prefix.
pub struct Greeter {
    prefix: String,
    seen: HashMap<String, u32>,
}

/// Supported locales.
pub enum Locale {
    En,
    Fr,
}

impl Greeter {
    /// Build a greeter with the default prefix.
    pub fn new() -> Self {
        Greeter { prefix: DEFAULT_GREETING.to_string(), seen: HashMap::new() }
    }

    /// Greet a single recipient.
    pub fn greet(&self, name: &str) -> String {
        format!("{}, {} ({})", self.prefix, name, LOCALE)
    }
}

/// Format a greeting via a greeter.
pub fn format_greeting(g: &Greeter, name: &str) -> String {
    g.greet(name)
}

/// A short alias for a list of recipient names.
pub type NameList = Vec<String>;

/// Raw bytes of a locale tag, either packed or split.
pub union LocaleBytes {
    packed: u32,
    parts: [u8; 4],
}

/// Behaviour shared by anything that can greet.
pub trait Greet {
    /// Greet a recipient by name (required).
    fn greet(&self, name: &str) -> String;

    /// Greet the world, with a default implementation.
    fn greet_world(&self) -> String {
        self.greet("world")
    }
}

/// Build the standard exclamatory greeting.
macro_rules! shout {
    ($name:expr) => {
        format!("Hello, {}!", $name)
    };
}

/// Locale helpers.
pub mod util {
    /// Strip surrounding whitespace from a locale tag.
    pub fn trim(tag: &str) -> &str {
        tag.trim()
    }
}
