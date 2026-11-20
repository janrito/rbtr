pub trait Saver {
    fn save(&self) -> bool {
        false
    }

    fn describe(&self) -> String {
        String::from("saver")
    }
}
