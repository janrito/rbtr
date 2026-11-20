use crate::saver::Saver;

pub struct Order;

impl Order {
    pub fn save(&self) -> bool {
        true
    }

    pub fn run(&self) -> String {
        self.save();
        self.describe()
    }
}

impl Saver for Order {}
