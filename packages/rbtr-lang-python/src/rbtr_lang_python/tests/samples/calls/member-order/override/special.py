from .order import Order


class Special(Order):
    def run(self):
        return self.save(), self.check()
