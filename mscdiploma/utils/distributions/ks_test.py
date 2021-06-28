from nirvana.utils.distributions.ecdf import ECDF
import numpy as np


class KSTest:
    """
    Kolmogorov-Smirnov similarity test
    """
    def __call__(self, ecdf0: ECDF, ecdf1: ECDF):
        assert isinstance(ecdf0, ECDF)
        assert isinstance(ecdf1, ECDF)
        X = np.concatenate((ecdf0.source_x, ecdf1.source_x))
        X.sort()
        Dn = 0
        for x in X:
            D = abs(ecdf0(x) - ecdf1(x))
            if D > Dn: Dn = D
        return Dn
