"""
Training configuration.

Everything the supervisor might want changed is a field here, not a literal
buried in the training loop. In particular `targets` is a list and `n_out` is
derived from it, so the unresolved D3 question (what the second output should
be) is answered by editing one line rather than the architecture.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# The three design inputs, in the order the paper lists them.
INPUTS = ["perveance_uperv", "rw_mm", "C"]

# Default targets. theta_cone_deg is the paper's own output and is not in
# question. Rc_mm is a PLACEHOLDER second output under reading C of D3 --
# see reports/DECISIONS.md. Reading B (the beam-envelope convergence half-angle
# at the anode) is the intended default, but it is computed by equation (II) of
# the physics model, which is behind the failed gate (D4). Changing this list
# is the whole change needed to switch readings.
DEFAULT_TARGETS = ["theta_cone_deg", "Rc_mm"]


@dataclass
class TrainConfig:
    # --- data ---
    targets: List[str] = field(default_factory=lambda: list(DEFAULT_TARGETS))
    inputs: List[str] = field(default_factory=lambda: list(INPUTS))
    beam_type: str = "pencil"           # standing rule 11; nothing else trains

    # Per-input transform, applied BEFORE min-max scaling. C is log-scaled
    # because ln C is the coordinate the physics is written in -- equation (II)
    # contains 0.5*ln(C) explicitly -- and because C spans a factor of 60, which
    # linear min-max compresses into the bottom sixth of the range. See D8.
    input_transform: Dict[str, str] = field(default_factory=lambda: {"C": "log"})

    # --- architecture (paper Table 1: 3 -> 3 -> 2 -> n_out, tansig throughout) ---
    hidden: List[int] = field(default_factory=lambda: [3, 2])
    activation: str = "tansig"
    output_activation: str = "tansig"   # {"tansig", "linear"}

    # --- residual weighting ---
    # Both targets are min-max scaled to [-1, 1] before residuals are formed, so
    # neither can dominate by magnitude and 1.0/1.0 is the principled default.
    # Exposed anyway because the moment a target is added on a different scale
    # (or the output activation goes linear) it stops being a no-op.
    residual_weights: Optional[List[float]] = None   # None -> all 1.0

    # --- Levenberg-Marquardt ---
    max_epochs: int = 5000
    mu_init: float = 1e-2
    mu_dec: float = 10.0
    mu_inc: float = 10.0
    mu_min: float = 1e-12
    mu_max: float = 1e10        # stop when damping reaches this: no step helps
    max_inner: int = 20         # mu increases per epoch before giving up
    grad_tol: float = 1e-10     # ||J^T e||
    dmse_tol: float = 1e-12     # improvement in training MSE

    # --- restarts ---
    restarts: int = 10
    seed: int = 100             # the notebook's seed, kept as the base

    # --- validation / early stopping ---
    # With 30 rows a held-out validation split is noise, so the default estimate
    # of generalisation inside the training rows is k-fold CV. See P5 notes.
    cv_folds: int = 5           # 0 disables; -1 means leave-one-out
    patience: int = 200         # early stopping on validation MSE, when one exists

    # --- diagnostics ---
    saturation_limit: float = 0.98   # warn beyond this in normalised output space

    @property
    def n_out(self) -> int:
        return len(self.targets)

    @property
    def n_in(self) -> int:
        return len(self.inputs)

    @property
    def layer_sizes(self) -> List[int]:
        return [self.n_in] + list(self.hidden) + [self.n_out]

    @property
    def weights(self) -> List[float]:
        if self.residual_weights is None:
            return [1.0] * self.n_out
        if len(self.residual_weights) != self.n_out:
            raise ValueError(
                f"residual_weights has {len(self.residual_weights)} entries "
                f"but there are {self.n_out} targets"
            )
        return list(self.residual_weights)
