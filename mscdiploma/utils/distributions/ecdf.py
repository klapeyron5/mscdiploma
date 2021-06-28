"""
https://en.wikipedia.org/wiki/Empirical_distribution_function
"""

import numpy as np


class StepFunction:
    def __init__(self, x, y, ival=0.0, sorted=False, side="left"):
        if side.lower() not in ["right", "left"]:
            msg = "side can take the values 'right' or 'left'"
            raise ValueError(msg)
        self.side = side
        _x = np.asarray(x)
        _y = np.asarray(y)
        if _x.shape != _y.shape:
            msg = "x and y do not have the same shape"
            raise ValueError(msg)
        if len(_x.shape) != 1:
            msg = "x and y must be 1-dimensional"
            raise ValueError(msg)
        self.x = np.r_[-np.inf, _x]
        self.y = np.r_[ival, _y]
        if not sorted:
            asort = np.argsort(self.x)
            self.x = np.take(self.x, asort, 0)
            self.y = np.take(self.y, asort, 0)
        self.n = self.x.shape[0]

    def __call__(self, time):
        tind = np.searchsorted(self.x, time, self.side) - 1
        return self.y[tind]


class ECDF(StepFunction):
    def __init__(self, x, side="right"):
        x = np.array(x, copy=True)
        x.sort()
        nobs = len(x)
        y = np.linspace(1.0 / nobs, 1, nobs)
        super(ECDF, self).__init__(x, y, side=side, sorted=True)
        self.source_x = x


def transform_arr_into_quantiles(x):
    assert type(x)==np.ndarray
    assert len(x.shape)==1

    F = ECDF(x)
    qx = [F(a) for a in x]
    return qx
