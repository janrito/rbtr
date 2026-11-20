pub trait Named {
    fn name(&self) -> String {
        String::from("named")
    }
}
