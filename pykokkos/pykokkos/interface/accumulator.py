from typing import Generic, TypeVar


class Acc(Generic[TypeVar("T")]):
    def __init__(self, val):
        self.val = val

    # Non-augmented arithmetic (`n + 1`, `n * 2`, ...) must not mutate `n`,
    # matching normal Python operator semantics.
    def __add__(self, other):
        return self.val + other

    def __radd__(self, other):
        return other + self.val

    def __sub__(self, other):
        return self.val - other

    def __rsub__(self, other):
        return other - self.val

    def __mul__(self, other):
        return self.val * other

    def __rmul__(self, other):
        return other * self.val

    def __truediv__(self, other):
        return self.val / other

    def __rtruediv__(self, other):
        return other / self.val

    def __floordiv__(self, other):
        return self.val // other

    def __rfloordiv__(self, other):
        return other // self.val

    def __mod__(self, other):
        return self.val % other

    def __rmod__(self, other):
        return other % self.val

    def __neg__(self):
        return -self.val

    # Augmented assignment (`n += 1`, ...) is where mutation belongs.
    def __iadd__(self, other):
        self.val = self.val + other
        return self

    def __isub__(self, other):
        self.val = self.val - other
        return self

    def __imul__(self, other):
        self.val = self.val * other
        return self

    def __itruediv__(self, other):
        self.val = self.val / other
        return self

    def __ifloordiv__(self, other):
        self.val = self.val // other
        return self

    def __imod__(self, other):
        self.val = self.val % other
        return self

    def __index__(self):
        return int(self.val)

    def not_(self):
        self.val = not self
        return self

    def lt(self, other):
        return self.val < other

    def le(self, other):
        return self.val <= other

    def eq(self, other):
        return self.val == other

    def ne(self, other):
        return self.val != other

    def ge(self, other):
        return self.val >= other

    def gt(self, other):
        return self.val > other
