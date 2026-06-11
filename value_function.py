import numpy as np


class ValueFunction:
    def __init__(self, T: int, ex_space, ey_space, etheta_space):
        self.T = T
        self.ex_space = ex_space
        self.ey_space = ey_space
        self.etheta_space = etheta_space

    def copy_from(self, other):
        raise NotImplementedError

    def update(self, t, ex, ey, etheta, target_value):
        raise NotImplementedError

    def __call__(self, t, ex, ey, etheta):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError


class GridValueFunction(ValueFunction):
    """
    Table-based: V[t, ix, iy, ith] stores the value at each discrete error state.
    """
    def __init__(self, T, ex_space, ey_space, etheta_space):
        super().__init__(T, ex_space, ey_space, etheta_space)
        self.V = np.zeros(
            (T, len(ex_space), len(ey_space), len(etheta_space)), dtype=np.float32
        )

    def copy_from(self, other):
        np.copyto(self.V, other.V)

    def update(self, t, ex, ey, etheta, target_value):
        # t, ex, ey, etheta may be scalars or arrays of grid indices
        self.V[t, ex, ey, etheta] = target_value

    def __call__(self, t, ex, ey, etheta):
        return self.V[t, ex, ey, etheta]

    def copy(self):
        new = GridValueFunction(self.T, self.ex_space, self.ey_space, self.etheta_space)
        new.V = self.V.copy()
        return new


class FeatureValueFunction(ValueFunction):
    """
    Feature-based value function (Optional Part 3) — not implemented.
    """
    pass
