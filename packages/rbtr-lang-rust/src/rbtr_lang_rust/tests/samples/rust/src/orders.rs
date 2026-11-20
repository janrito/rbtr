//! Orders — calls through `Self`, `self`, a type, and a `use` binding.

use crate::config::LOCALE as locale;

pub struct Order {
    name: String,
}

impl Order {
    pub fn new(name: &str) -> Self {
        Self { name: name.to_string() }
    }

    pub fn build() -> Order {
        let order = Self::new(locale);
        order.check();
        Order::new("x")
    }

    fn check(&self) -> bool {
        self.valid()
    }

    fn valid(&self) -> bool {
        true
    }
}
