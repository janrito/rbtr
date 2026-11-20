from .left import Left
from .right import Right


class Bottom(Left, Right):
    def run(self):
        return self.m()
