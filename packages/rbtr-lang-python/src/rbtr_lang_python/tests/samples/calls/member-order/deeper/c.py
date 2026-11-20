from .a import A
from .b import B


class C(A, B):
    """The deeper case."""

    def run(self):
        return self.m()
