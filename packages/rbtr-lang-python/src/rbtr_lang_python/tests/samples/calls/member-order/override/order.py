from .base import Base


class Order(Base):
    def save(self):
        return super().save()
