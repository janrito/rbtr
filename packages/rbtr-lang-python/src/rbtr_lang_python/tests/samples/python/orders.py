"""Orders — calls of each shape the call graph resolves, and two it leaves."""

from . import python as greetings
from .config import LOCALE as locale_name
from .python import *


class Model:
    """A base whose members a subclass reaches through ``self`` and ``super()``."""

    def save(self) -> bool:
        return self.check()

    def check(self) -> bool:
        return True


class Order(Model):
    """A subclass calling its own members, its base's, and other modules'."""

    def save(self) -> bool:
        super().save()
        return bool(greetings.Greeter().greet(locale_name))

    def total(self, items: list[int]) -> int:
        # `sum` is defined nowhere here, and `items[0]` is a value: no edge.
        return sum(items) + int(self.check()) + items[0].bit_length()


def register(name: str):
    def wrap(fn):
        return fn

    return wrap


@register("orders")
def build() -> Order:
    def inner() -> Order:
        return Order()

    return inner()


format_greeting("order")
