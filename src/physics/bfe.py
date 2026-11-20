"""
Beam-focusing electrode (BFE) and anode shapes of a convergent Pierce gun.

Tiwary U., Basu B.N., "Noniterative synthesis of convergent Pierce electron
guns by the conformal mapping of electrode shapes using logarithmic
transformation", IEEE Trans. Electron Devices 35(7) (1988) 1184-1187.

The idea (their Secs. II-III). Inside the beam the potential is Langmuir and
Blodgett's spherical-flow solution; on the beam edge theta = theta0,

    V(rho) = C * alpha(gamma)^(4/3),   gamma = ln(Rc / rho),

alpha being the same series Vaughan's Eq.(5) uses (a1..a6, Vaughan's improved
last coefficient). The logarithmic map w = ln z sends the cathode and anode
spheres to straight lines and the beam edge to a straight line, so the outside
potential that matches the beam edge is the real part of the same function of
the complex variable

    s = ln(rho / Rc) + j (theta - theta0),    V = Re{ C * alpha(-s)^(4/3) }.

The V = 0 equipotential is the BFE; the V = V_a equipotential is the anode.

How it is computed here. The paper finds the equipotentials by Newton forward
interpolation along radial lines, an approximation it estimates at 0.26-0.41 %
error in the anode radius. Here they are traced exactly instead, by inverting
the series in the complex plane:

    BFE   :  alpha(-s) = t * exp(-j 3pi/8),           t >= 0
    anode :  alpha(-s) = (K + j tau)^(3/4),            tau <= 0,  K = alpha(y)^(4/3)

since Re{(t e^{-j3pi/8})^(4/3)} = Re{t^(4/3) e^{-j pi/2}} = 0 and
Re{(K + j tau)} = K. The first form makes the BFE leave the cathode rim at
exactly 3pi/8 = 67.5 deg to the beam edge (Pierce's angle), and shows the
paper's remark that the BFE shape in (ln rho/Rc, theta - theta0) does not
depend on the gun at all. The second passes through the anode sphere radius Ra
on the beam edge exactly.

What is approximate, and stated rather than hidden (paper Sec. IV): the map is
conformal in the 2-D (ln rho, theta) plane, so V satisfies the 2-D Laplace
equation there, while the axisymmetric 3-D Laplacian has two extra
first-derivative terms. `laplace_residual` measures that. The anode aperture
and thermal velocities are ignored, as in the paper. The PIERCE simulator
(src/sim) is the full check.

Geometry convention: polar (rho, theta) about the common centre of the cathode
and anode spheres, theta from the axis; with the cathode vertex at z = d,
z = d + Rc - rho cos(theta), r = rho sin(theta). Rc, Ra, theta0 are the ideal
flow's (Vaughan's uncorrected theta_raw), which is what the potential belongs to.
"""

import math

import numpy as np

from . import lb

_A = (1.0,) + lb._A                     # alpha = sum A[k] g^(k+1)
PIERCE = 3 * math.pi / 8                # 67.5 deg


def alpha(g):
    """Vaughan Eq.(5) series, valid for complex gamma."""
    g = np.asarray(g, dtype=complex)
    out = np.zeros_like(g)
    for c in reversed(_A):
        out = (out + c) * g
    return out


def dalpha(g):
    g = np.asarray(g, dtype=complex)
    out = np.zeros_like(g)
    for k in range(len(_A) - 1, -1, -1):
        out = out * g + (k + 1) * _A[k]
    return out


def alpha_inv(a, g0, n_iter=60, tol=1e-14):
    """Complex Newton inversion of alpha(g) = a starting from g0."""
    g = complex(g0)
    for _ in range(n_iter):
        step = (complex(alpha(g)) - a) / complex(dalpha(g))
        g -= step
        if abs(step) < tol * max(1.0, abs(g)):
            return g
    raise ArithmeticError(f"alpha inversion did not converge at a = {a}")


GAMMA_MAX = 2.9          # Eq.(5) series held to ~1 % up to -alpha ~ 9.6 (gamma ~ 2.9)


def _trace(target, params, g_start):
    """Follow alpha(-s) = target(p) along params; returns s values."""
    out, g = [], g_start
    for p in params:
        g = alpha_inv(target(p), g)
        if abs(g) > GAMMA_MAX:
            break
        out.append(-g)
    return np.array(out)


def bfe_s(t_max=6.0, n=400):
    """The BFE in the s-plane. Universal: no gun parameter enters."""
    ts = np.linspace(0.0, t_max, n + 1)[1:]
    s = _trace(lambda t: t * np.exp(-1j * PIERCE), ts, ts[0] * np.exp(-1j * PIERCE))
    return np.concatenate([[0j], s])


def anode_s(y, tau_max=40.0, n=400):
    """The anode equipotential in the s-plane for y = ln(Rc/Ra)."""
    K = complex(alpha(y)) ** (4.0 / 3.0)
    taus = -np.linspace(0.0, tau_max, n + 1)
    return _trace(lambda tau: (K + 1j * tau) ** 0.75, taus, complex(y))


def equipotential_s(p, y, tau_max=40.0, n=400):
    """Any equipotential V = p * V_a outside the beam (0 < p <= 1)."""
    K = complex(alpha(y)) ** (4.0 / 3.0)
    g0 = alpha_inv(complex((p * K.real) ** 0.75), complex(y) * p)
    taus = -np.linspace(0.0, tau_max, n + 1)
    return _trace(lambda tau: (p * K + 1j * tau) ** 0.75, taus, g0)


def to_gun(s, Rc, theta0, d=0.0):
    """s-plane points -> (z, r) in mm, cathode vertex at z = d."""
    rho = Rc * np.exp(s.real)
    th = theta0 + s.imag
    return np.column_stack([d + Rc - rho * np.cos(th), rho * np.sin(th)])


def electrodes(Rc, Ra, theta0_deg, d=0.0, r_max=None):
    """
    BFE and anode curves for a gun, in mm, cathode vertex at z = d.

    Rc, Ra: cathode and anode sphere radii; theta0_deg: the half-angle of the
    ideal flow (theta_raw). Curves are cut where they leave r <= r_max
    (default 1.75 x the cathode disc radius, Vedant's outer wall).
    Returns dict(bfe=(N,2), anode=(M,2), rim=(z, r), anode_axis_point=(z, r)).
    """
    th0 = math.radians(theta0_deg)
    y = math.log(Rc / Ra)
    rc_disc = Rc * math.sin(th0)
    r_max = 1.75 * rc_disc if r_max is None else r_max
    cut = lambda P: P[: np.argmax(P[:, 1] > r_max)] if (P[:, 1] > r_max).any() else P
    bfe = cut(to_gun(bfe_s(), Rc, th0, d))
    an = cut(to_gun(anode_s(y), Rc, th0, d))
    return {"bfe": bfe, "anode": an, "rim": (d + Rc * (1 - math.cos(th0)), rc_disc),
            "y": y, "theta0": th0}


def start_angle_deg(Rc, theta0_deg, k=3):
    """Angle between the BFE's first chord and the beam edge (should be 67.5)."""
    th0 = math.radians(theta0_deg)
    P = to_gun(bfe_s(t_max=1e-3, n=50), Rc, th0)
    v = P[k] - P[0]
    edge = np.array([math.cos(th0), -math.sin(th0)])     # rim -> centre, along the beam edge
    return math.degrees(math.acos(np.dot(v, edge) / np.linalg.norm(v)))


def potential(rho, theta, Rc, Ra, theta0):
    """Normalised potential V/V_a outside the beam (theta >= theta0)."""
    s = np.log(rho / Rc) + 1j * (theta - theta0)
    K = complex(alpha(math.log(Rc / Ra))) ** (4.0 / 3.0)
    return (alpha(-s) ** (4.0 / 3.0)).real / K.real


def laplace_residual(Rc, Ra, theta0_deg, n=48, span_deg=40.0):
    """
    How far the conformal-map potential is from the true axisymmetric Laplace
    equation, between the cathode and anode spheres just outside the beam.

    With u = ln rho, the spherical Laplacian is
        rho^2 lap V = V_uu + V_u + V_tt + cot(t) V_t,
    and the map makes V_uu + V_tt = 0 exactly, so the residual is
    V_u + cot(t) V_t. Reported relative to |V_uu| + |V_tt| (the size of the
    terms that do cancel), as max and RMS over the mesh.
    """
    th0 = math.radians(theta0_deg)
    us = np.linspace(math.log(Ra), math.log(Rc), n)[1:-1]
    ts = th0 + np.radians(np.linspace(0.5, span_deg, n))[1:-1]
    U, T = np.meshgrid(us, ts, indexing="ij")
    h = 1e-4
    f = lambda u, t: potential(np.exp(u), t, Rc, Ra, th0)
    V0 = f(U, T)
    Vu = (f(U + h, T) - f(U - h, T)) / (2 * h)
    Vt = (f(U, T + h) - f(U, T - h)) / (2 * h)
    Vuu = (f(U + h, T) - 2 * V0 + f(U - h, T)) / h ** 2
    Vtt = (f(U, T + h) - 2 * V0 + f(U, T - h)) / h ** 2
    rel = np.abs(Vu + Vt / np.tan(T)) / (np.abs(Vuu) + np.abs(Vtt))
    harmonic_2d = np.abs(Vuu + Vtt) / (np.abs(Vuu) + np.abs(Vtt))
    return {"max": float(rel.max()), "rms": float(np.sqrt((rel ** 2).mean())),
            "check_2d_harmonic_max": float(harmonic_2d.max())}


def from_inputs(P_uperv, rw_mm, C, theta0_deg, n=4000):
    """
    Tiwary & Basu's electrodes straight from the ANN's inputs and angle.

    Their synthesis starts from the cathode radius, the ratio rc/ra and the
    half-beam cone angle theta0 (their Sec. IV, from their ref. [1]). From the
    Predict tab's inputs these are: r_c = sqrt(C) r_w (disc radius),
    Rc = r_c / sin(theta0) (cathode sphere), and Ra from the Langmuir-Blodgett
    cone relation P = 14.665 (1 - cos theta0) / alpha^2 (the cone constant,
    src/physics/lb.py), alpha -> y = ln(Rc/Ra) by Vaughan's series. Then the
    focus electrode (V = 0) and anode face (V = Va) as above, each cut at its
    highest point. Cathode vertex at z = 0.

    Returns dict with rc, Rc, Ra, y, the two curves (z, r) in mm, and the
    electrode's angle to the beam edge (67.5 deg, the paper's result) and to the
    axis (67.5 - theta0) where it leaves the cathode rim.
    """
    th = math.radians(theta0_deg)
    rc = math.sqrt(C) * rw_mm
    Rc = rc / math.sin(th)
    a = math.sqrt(lb.CONE_CONST * (1 - math.cos(th)) / P_uperv)
    y = float(lb.y_from_a(a))
    Ra = Rc * math.exp(-y)
    up = lambda P: P[: int(np.argmax(P[:, 1])) + 1]
    B = up(to_gun(bfe_s(n=n), Rc, th))
    A = up(to_gun(anode_s(y, n=n), Rc, th))
    return {"rc": rc, "Rc": Rc, "Ra": Ra, "y": y, "alpha": a, "bfe": B, "anode": A,
            "angle_to_edge_deg": math.degrees(PIERCE),
            "angle_to_axis_deg": math.degrees(PIERCE) - theta0_deg,
            "bfe_tip": tuple(B[-1]), "anode_top": tuple(A[-1]),
            "anode_on_axis_z": Rc - Ra}
