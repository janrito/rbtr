use crate::named::Named;

pub trait Titled: Named {
    fn title(&self) -> String {
        self.name()
    }
}
