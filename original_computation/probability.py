import warnings
import numpy as np
from numba import jit
from scipy.integrate import quad

# Suppress only IntegrationWarning
warnings.filterwarnings("ignore")


# @jit(nopython=True)
# def integrand(tau, x1, x2, kr, M):
#     p = np.exp(-x2 * kr * (x1 - tau))
#     integrand_value = np.exp(tau) * p * ((1 - p) ** (M - 1))
#     return integrand_value


# def probability(x1, x2, kr, M):
#     factor_up_front = 1.0 / (np.exp(x1) - 1)
#     integral = quad(
#         integrand,
#         0,
#         x1,
#         args=(x1, x2, kr, M),
#         limit=200,
#         epsabs=1e-10,
#         epsrel=1e-10,
#     )[0]
#     return factor_up_front * integral


@jit(nopython=True)
def integrand(x, xw, xp, ri, Mi):
    p = np.exp(-xp * ri * (xw - x))
    integrand_value = np.exp(x) * p * ((1 - p) ** (Mi - 1))
    return integrand_value


def probability(xw, xp, ri, Mi):
    factor_up_front = 1.0 / (np.exp(xw) - 1)
    integral = quad(
        integrand,
        0,
        xw,
        args=(xw, xp, ri, Mi),
        limit=200,
        epsabs=1e-10,
        epsrel=1e-10,
    )[0]
    return factor_up_front * integral
