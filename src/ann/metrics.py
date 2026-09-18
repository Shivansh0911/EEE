"""
Error measures, per Panahi et al. eqs (8)-(11).

All of these are reported in PHYSICAL units (degrees, millimetres), not in the
normalised [-1, 1] space the network trains in. The paper's numbers are physical
and a comparison against them has to be like for like -- a normalised RMSE would
look flattering and mean nothing.

    MRE  = (100/N) sum |y - yhat| / |y|        mean relative error, per cent
    MSE  = (1/N) sum (y - yhat)^2
    RMSE = sqrt(MSE)
    R    = Pearson correlation between y and yhat
"""

import numpy as np


def _cols(Y):
    Y = np.asarray(Y, dtype=float)
    return Y if Y.ndim == 2 else Y.reshape(-1, 1)


def mre(y, yhat):
    """Mean relative error, per cent. Eq. (8)."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    if np.any(y == 0):
        raise ValueError("MRE is undefined where the true value is zero")
    return 100.0 * float(np.mean(np.abs(y - yhat) / np.abs(y)))


def mse(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.mean((y - yhat) ** 2))


def rmse(y, yhat):
    return float(np.sqrt(mse(y, yhat)))


def pearson_r(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    if y.size < 2:
        return float("nan")
    sy, sp = y.std(), yhat.std()
    if sy == 0 or sp == 0:
        return float("nan")
    return float(np.mean((y - y.mean()) * (yhat - yhat.mean())) / (sy * sp))


def per_output(Y, Yhat, names):
    """{target_name: {mre, mse, rmse, r, max_abs_err}} in physical units."""
    Y, Yhat = _cols(Y), _cols(Yhat)
    out = {}
    for k, name in enumerate(names):
        y, p = Y[:, k], Yhat[:, k]
        out[name] = {
            "mre_pct": mre(y, p),
            "mse": mse(y, p),
            "rmse": rmse(y, p),
            "pearson_r": pearson_r(y, p),
            "max_abs_err": float(np.max(np.abs(y - p))),
            "n": int(y.size),
        }
    return out


def format_table(blocks, names):
    """
    blocks: {"train": per_output(...), "test": per_output(...)}
    Returns a markdown table, one row per (split, target).
    """
    lines = [
        "| Split | Target | n | MRE % | MSE | RMSE | Pearson R | Max abs err |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for split, block in blocks.items():
        for name in names:
            m = block[name]
            lines.append(
                f"| {split} | `{name}` | {m['n']} | {m['mre_pct']:.2f} | "
                f"{m['mse']:.4f} | {m['rmse']:.4f} | {m['pearson_r']:.4f} | "
                f"{m['max_abs_err']:.4f} |"
            )
    return "\n".join(lines)
