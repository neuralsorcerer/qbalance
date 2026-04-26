# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from qbalance.strategies import StrategySpec


def _featurize(spec: StrategySpec) -> np.ndarray:
    # One-hot-ish hand features

    """Internal helper that featurize.

    Args:
        spec: Strategy/backend specification controlling compilation behavior.

    Returns:
        np.ndarray with the computed result.

    Raises:
        None.
    """
    return np.asarray(
        [
            1.0,
            float(spec.optimization_level),
            1.0 if spec.routing_method == "sabre" else 0.0,
            1.0 if spec.layout_method == "sabre" else 0.0,
            1.0 if spec.layout_method == "qbalance_noise_aware" else 0.0,
            1.0 if spec.pauli_twirling else 0.0,
            float(spec.num_twirls) if spec.pauli_twirling else 0.0,
            1.0 if spec.dynamical_decoupling else 0.0,
            1.0 if spec.measurement_twirling else 0.0,
            1.0 if spec.mthree else 0.0,
            1.0 if spec.zne else 0.0,
            1.0 if spec.cutting else 0.0,
        ],
        dtype=float,
    )


@dataclass
class BanditSearcher:
    """Thompson-sampling style search over candidate strategies.

    We keep a Bayesian linear regression surrogate over feature vectors and sample coefficients.
    """

    alpha: float = 1.0
    sigma2: float = 1.0

    def __post_init__(self) -> None:
        """Validate and normalize dataclass state immediately after initialization.

        Args:
            None.

        Returns:
            None. This method updates state or performs side effects only.

        Raises:
            ValueError: Raised when input validation fails or a dependent operation cannot be completed.
        """
        if not math.isfinite(self.alpha) or self.alpha <= 0.0:
            raise ValueError("alpha must be a finite positive value")
        if not math.isfinite(self.sigma2) or self.sigma2 <= 0.0:
            raise ValueError("sigma2 must be a finite positive value")

        self._feature_dim = len(_featurize(StrategySpec()))
        self._X: List[np.ndarray] = []
        self._y: List[float] = []

    def observe(self, spec: StrategySpec, score: float) -> None:
        """Observe used by the qbalance workflow.

        Args:
            spec: Strategy/backend specification controlling compilation behavior.
            score: Score value consumed by this routine.

        Returns:
            None. This method updates state or performs side effects only.

        Raises:
            ValueError: Raised when input validation fails or a dependent operation cannot be completed.
        """
        score_value = float(score)
        if not math.isfinite(score_value):
            raise ValueError("score must be finite")

        self._X.append(_featurize(spec))
        self._y.append(score_value)

    def _posterior(self) -> Tuple[np.ndarray, np.ndarray]:
        """Internal helper that posterior.

        Args:
            None.

        Returns:
            Tuple[np.ndarray, np.ndarray] with the computed result.

        Raises:
            None.
        """
        if not self._X:
            mean = np.zeros(self._feature_dim)
            precision = np.eye(self._feature_dim) * self.alpha
            return mean, precision

        X = np.vstack(self._X)
        y = np.asarray(self._y)
        # Ridge posterior precision (inverse covariance).
        precision = self.alpha * np.eye(X.shape[1]) + (X.T @ X) / self.sigma2
        rhs = (X.T @ y) / self.sigma2
        mean = np.linalg.solve(precision, rhs)
        return mean, precision

    def propose(
        self, candidates: Sequence[StrategySpec], rng: np.random.Generator
    ) -> StrategySpec:
        """Propose used by the qbalance workflow.

        Args:
            candidates: Candidate strategies considered during selection.
            rng: NumPy random generator used for stochastic selection.

        Returns:
            StrategySpec with the computed result.

        Raises:
            ValueError: Raised when input validation fails or a dependent operation cannot be completed.
        """
        if not candidates:
            raise ValueError("candidates must contain at least one strategy")

        mean, precision = self._posterior()
        # Sample without materializing covariance: if precision = L L^T and
        # z ~ N(0, I), then mean + solve(L^T, z) ~ N(mean, precision^{-1}).
        chol = np.linalg.cholesky(precision)
        z = rng.standard_normal(self._feature_dim)
        w = mean + np.linalg.solve(chol.T, z)

        features = np.vstack([_featurize(c) for c in candidates])
        scores = features @ w
        best_idx = int(np.argmin(scores))
        return candidates[best_idx]
