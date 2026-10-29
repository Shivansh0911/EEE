"""
Langmuir-Blodgett geometry of the converging spherical diode, in float64.

Every constant here is either derived from CODATA physical constants (and the
derivation is the code) or is an empirical coefficient quoted with its source.

The constant that went wrong (D10)
----------------------------------
Langmuir I., Blodgett K.B., "Currents limited by space charge between
concentric spheres", Phys. Rev. 24 (1924) 49-59, give the space-charge-limited
current between FULL concentric spheres as

    I = (16 pi / 9) eps0 sqrt(2e/m) V^1.5 / alpha^2,

i.e. a full-sphere microperveance constant of 29.33. A Pierce gun uses only the
cone of half-angle theta, which subtends the fraction (1 - cos theta) / 2 of the
sphere, so the cone constant is HALF of that:

    muP = 14.665 (1 - cos theta) / alpha^2.

Vaughan J.R.M., "Synthesis of the Pierce gun", IEEE Trans. ED-28(1) (1981) 37,
Eq.(3), prints 14.67 -- the same constant, rounded. Our earlier probe
(`probe_closure.py`) used 29.33 with (1 - cos theta), a factor of two too large,
and the fitted lens coefficient c = 0.62 (instead of Vaughan's 0.417) absorbed
part of that error. `CONE_CONST` is the exact value; `VAUGHAN_CONE_CONST` is the
printed one that Vedant's solver uses, and the emulator uses it too so that it
reproduces his solver rather than a slightly different one.
"""

import math

import numpy as np

# --- CODATA 2018 --------------------------------------------------------------
EPS0 = 8.8541878128e-12        # F/m
E_CHARGE = 1.602176634e-19     # C (exact)
M_E = 9.1093837015e-31         # kg
ETA = E_CHARGE / M_E           # e/m

# Full-sphere Langmuir-Blodgett constant, microperveance units.
SPHERE_CONST = (16.0 * math.pi / 9.0) * EPS0 * math.sqrt(2.0 * ETA) * 1e6   # 29.33
# The cone of half-angle theta: (1 - cos theta)/2 of the sphere.
CONE_CONST = SPHERE_CONST / 2.0                                              # 14.665
# Vaughan (1981) Eq.(3) as printed, and as used by the vendored solver.
VAUGHAN_CONE_CONST = 14.67

# Paraxial space-charge spreading after the anode, r'' = K/r with
# K = I / (4 pi eps0 sqrt(2 eta) V^1.5), integrates to r' = c sqrt(muP ln(r/rw)),
# c = sqrt(2e-6 / (4 pi eps0 sqrt(2 eta))). Vaughan (1981) Eq.(9)/(14): 0.17409.
SPREAD_CONST = math.sqrt(2e-6 / (4.0 * math.pi * EPS0 * math.sqrt(2.0 * ETA)))

# Empirical coefficients, quoted, not fitted here.
LENS_COEFF = 0.417        # Vaughan (1981) Eq.(10), anode-aperture lens
ABERRATION_K = 0.905      # Vaughan (1981) Sec. III-F, sin(theta)/sin(theta_T)
APERTURE_FACTOR = 1.2     # Vaughan (1981) Eq.(18), ra = 1.2 rb(za)
THETA_ZA_MAX_DEG = 56.44  # above this Eq.(20) has no real root at factor 1.2
A_MAX = 9.6               # Eq.(5) series valid to 1 % up to -alpha ~ 9.6

# Eq.(5) forward series (-alpha) = y + 0.3y^2 + 0.075y^3 + 0.01432y^4
# + 0.00216y^5 + 0.00035y^6, y = ln(Rc/Ra). The last coefficient is Vaughan's
# improved 0.00035 (the classical value is 0.00027).
_A = (0.3, 0.075, 0.01432, 0.00216, 0.00035)
# d(-alpha)/dy, used as the Newton derivative and, with two coefficients
# changed, as the P-polynomial of Eq.(10). Both exactly as in the vendored code.
_D = (0.6, 0.225, 0.05728, 0.0108, 0.00162)
_P = (0.6, 0.225, 0.0573, 0.0108, 0.0021)
_Y0 = (0.275, 0.06, 0.006)   # Eq.(6) approximate inverse, a Newton start


def a_from_y(y):
    """Vaughan (1981) Eq.(5): (-alpha) from y = ln(Rc/Ra). Horner form."""
    y = np.asarray(y, dtype=float)
    return y + y * y * (_A[0] + y * (_A[1] + y * (_A[2] + y * (_A[3] + y * _A[4]))))


def da_dy(y):
    """Derivative of Eq.(5), as the vendored solver writes it."""
    return 1 + y * (_D[0] + y * (_D[1] + y * (_D[2] + y * (_D[3] + y * _D[4]))))


def p_poly(y):
    """Vaughan (1981) Eq.(10) polynomial P(y), as in the vendored solver."""
    return 1 + y * (_P[0] + y * (_P[1] + y * (_P[2] + y * (_P[3] + y * _P[4]))))


def y_from_a(a, n_iter=20):
    """Invert Eq.(5) by Newton from the Eq.(6) start. Vectorised.

    Runs a fixed number of iterations rather than stopping at a tolerance, so
    the result is the same function in every element of a vector (and in the
    JavaScript port); Newton is quadratic, so 20 is far past convergence.
    """
    a = np.asarray(a, dtype=float)
    y = a - _Y0[0] * a**2 + _Y0[1] * a**3 - _Y0[2] * a**4
    for _ in range(n_iter):
        y = y - (a_from_y(y) - a) / da_dy(y)
    return y


def cone_perveance(theta_rad, alpha_neg, const=CONE_CONST, kf=1.0):
    """Vaughan (1981) Eq.(3): muP = const * kf * (1 - cos theta) / (-alpha)^2."""
    return const * kf * (1.0 - np.cos(theta_rad)) / np.asarray(alpha_neg) ** 2
