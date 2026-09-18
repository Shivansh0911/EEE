"""
Levenberg-Marquardt training.

With 26 parameters this is a small non-linear least-squares problem, which is
precisely where LM is the right tool. Gradient descent is robust but slow and
zig-zags along narrow valleys; Gauss-Newton approximates the error surface as a
quadratic bowl and jumps to its minimum, which is fast near a solution and
divergent far from one. LM blends the two through a damping parameter mu:

    (J^T J + mu I) delta = -J^T e

Large mu -> mu*I dominates -> a small step along -J^T e -> gradient descent, safe.
Small mu -> reduces to Gauss-Newton -> big well-aimed jumps, fast. And mu adapts
itself: a step that improved the loss is accepted and mu is divided by 10 (be
bolder); a step that made it worse is rejected and mu multiplied by 10 (be
cautious). There is no learning rate to tune. This is MATLAB's `trainlm`, which
is what the paper used.

It does not scale -- J^T J is 26x26 here but would be millions-squared for a
deep network, which is why modern deep learning uses Adam/SGD instead.

Two structural fixes over the notebook's loop:

1.  The Jacobian is recomputed ONLY after an ACCEPTED step. A rejected step
    leaves the parameters unchanged, so the Jacobian at those parameters is
    still exactly valid; recomputing it was pure waste. Rejections are retried
    in an inner loop with larger mu against the Jacobian already in hand.
2.  Real stopping criteria, rather than always running the full epoch budget:
    mu past mu_max (no amount of damping helps any more), ||J^T e|| below
    grad_tol (a stationary point), or the improvement in MSE below dmse_tol.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .jacobian import analytic_jacobian
from .network import mse as mse_of, nguyen_widrow, residuals


@dataclass
class TrainResult:
    theta: np.ndarray                       # best parameters found
    train_mse: float                        # normalised-space MSE at `theta`
    val_mse: Optional[float]
    history: List[float] = field(default_factory=list)       # train MSE per epoch
    val_history: List[float] = field(default_factory=list)   # val MSE per epoch
    epochs_run: int = 0
    best_epoch: int = 0
    accepted_steps: int = 0
    rejected_steps: int = 0
    stop_reason: str = ""
    seed: Optional[int] = None


def train_lm(X, Y, cfg, theta0=None, val=None, rng=None, seed=None, verbose=False):
    """
    Fit one model. X, Y are NORMALISED.

    `val` is an optional (X_val, Y_val) pair, also normalised. When present,
    training tracks validation MSE, keeps the parameters at its minimum, and
    stops early after `cfg.patience` epochs without improvement. When absent
    the best-by-training-MSE parameters are returned and there is no early
    stopping -- which is the honest situation with the paper's 23/7 split, where
    there is no room for a validation set.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    theta = nguyen_widrow(cfg.layer_sizes, rng) if theta0 is None else np.array(theta0, float)

    Xv = Yv = None
    if val is not None:
        Xv, Yv = val

    mu = cfg.mu_init
    e = residuals(theta, X, Y, cfg)
    f = float(np.mean(e * e))
    J = analytic_jacobian(theta, X, Y, cfg)
    eye = np.eye(theta.size)

    res = TrainResult(theta=theta.copy(), train_mse=f, val_mse=None, seed=seed)
    if Xv is not None:
        res.val_mse = mse_of(theta, Xv, Yv, cfg)

    best_score = res.val_mse if Xv is not None else f
    stale = 0
    stop = "max_epochs"

    for epoch in range(cfg.max_epochs):
        grad = J.T @ e
        if np.linalg.norm(grad) < cfg.grad_tol:
            stop = "grad_tol"
            break

        JtJ = J.T @ J
        accepted = False

        # Inner loop: the Jacobian is fixed here. Only mu changes.
        for _ in range(cfg.max_inner):
            try:
                delta = -np.linalg.solve(JtJ + mu * eye, grad)
            except np.linalg.LinAlgError:
                mu = min(mu * cfg.mu_inc, cfg.mu_max * 10)
                res.rejected_steps += 1
                if mu > cfg.mu_max:
                    break
                continue

            cand = theta + delta
            e_new = residuals(cand, X, Y, cfg)
            f_new = float(np.mean(e_new * e_new))

            if f_new < f:
                df = f - f_new
                theta, e, f = cand, e_new, f_new
                mu = max(mu / cfg.mu_dec, cfg.mu_min)
                res.accepted_steps += 1
                accepted = True
                # Only now is the old Jacobian stale.
                J = analytic_jacobian(theta, X, Y, cfg)
                break

            mu = min(mu * cfg.mu_inc, cfg.mu_max * 10)
            res.rejected_steps += 1
            if mu > cfg.mu_max:
                break

        res.history.append(f)
        if Xv is not None:
            fv = mse_of(theta, Xv, Yv, cfg)
            res.val_history.append(fv)

        score = res.val_history[-1] if Xv is not None else f
        if score < best_score - 0.0:
            best_score = score
            res.theta = theta.copy()
            res.best_epoch = epoch
            res.train_mse = f
            res.val_mse = res.val_history[-1] if Xv is not None else None
            stale = 0
        else:
            stale += 1

        if verbose and epoch % 50 == 0:
            tail = f" | val {res.val_history[-1]:.3e}" if Xv is not None else ""
            print(f"  epoch {epoch:4d} | mse {f:.6e} | mu {mu:.2e}{tail}")

        if not accepted:
            stop = "mu_max"
            break
        if mu > cfg.mu_max:
            stop = "mu_max"
            break
        if df < cfg.dmse_tol:
            stop = "dmse_tol"
            break
        if Xv is not None and stale >= cfg.patience:
            stop = "early_stopping"
            break

    res.epochs_run = len(res.history)
    res.stop_reason = stop

    # If there was no validation set, "best" means best training MSE, and the
    # final parameters are that by construction of the accept test.
    if Xv is None:
        res.theta = theta.copy()
        res.train_mse = f
    return res


def train_with_restarts(X, Y, cfg, val=None, score_fn=None, verbose=False):
    """
    Run `cfg.restarts` Nguyen-Widrow initialisations and keep the best.

    A single seed means the result is partly luck: LM finds a local minimum, and
    which one depends entirely on where it started. Ten restarts turn that
    lottery into a choice.

    Selection uses, in order of preference: an explicit `score_fn(result)`, the
    validation MSE when a validation set exists, and otherwise the training MSE.
    That last case is selection on the training set and is weaker -- it is
    reported as such rather than dressed up.
    """
    results = []
    for i in range(cfg.restarts):
        seed = cfg.seed + i
        r = train_lm(X, Y, cfg, val=val, seed=seed, verbose=verbose)
        if score_fn is not None:
            r_score = score_fn(r)
        elif r.val_mse is not None:
            r_score = r.val_mse
        else:
            r_score = r.train_mse
        results.append((r_score, i, r))

    results.sort(key=lambda t: (t[0], t[1]))
    best_score, _, best = results[0]
    return best, [(s, r) for s, _, r in results], best_score
