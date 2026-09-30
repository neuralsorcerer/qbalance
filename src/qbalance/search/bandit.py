# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from qbalance.strategies import StrategySpec


def _featurize(spec: StrategySpec) -> np.ndarray:
    """Return the surrogate's feature vector of ``spec``.

    A constant, the optimization level, indicators for SABRE routing, SABRE
    layout, the noise-aware layout, Pauli twirling, dynamical decoupling,
    measurement twirling, M3, ZNE and cutting, and ``num_twirls`` when the
    strategy twirls (0 otherwise).
    """
    return np.asarray(
        [
            1.0,
            float(spec.optimization_level),
            1.0 if spec.routing_method == "sabre" else 0.0,
            1.0 if spec.layout_method == "sabre" else 0.0,
            1.0 if spec.layout_method == "qbalance_noise_aware" else 0.0,
            1.0 if spec.pauli_twirling else 0.0,
            # num_twirls is the ensemble size for Pauli and measurement twirling.
            (
                float(spec.num_twirls)
                if spec.pauli_twirling or spec.measurement_twirling
                else 0.0
            ),
            1.0 if spec.dynamical_decoupling else 0.0,
            1.0 if spec.measurement_twirling else 0.0,
            1.0 if spec.mthree else 0.0,
            1.0 if spec.zne else 0.0,
            1.0 if spec.cutting else 0.0,
        ],
        dtype=float,
    )


# Observation-noise variance assumed before two scores have been observed.
_PRIOR_NOISE_VARIANCE = 1.0
_MIN_NOISE_VARIANCE = 1e-6


@dataclass
class BanditSearcher:
    """Thompson-sampling search over candidate strategies.

    A Bayesian linear regression of a candidate's score on its features
    (see :func:`_featurize`), with prior ``w ~ N(0, I / alpha)`` and
    observation noise of variance ``sigma2``; each proposal samples ``w``
    from the posterior and picks the candidate it scores lowest.

    Args:
        alpha: Prior precision of the feature weights.
        sigma2: Observation-noise variance.  ``None`` matches it to the sample
            variance of the scores observed so far, which keeps exploration
            calibrated whatever their scale; a fixed value is only right for
            scores on a known scale.
    """

    alpha: float = 1.0
    sigma2: Optional[float] = 1.0

    def __post_init__(self) -> None:
        """Validate the hyperparameters and start with no observations.

        Raises:
            ValueError: If ``alpha`` is not finite and positive, or ``sigma2``
                is neither ``None`` nor finite and positive.
        """
        if not math.isfinite(self.alpha) or self.alpha <= 0.0:
            raise ValueError("alpha must be a finite positive value")
        if self.sigma2 is not None and (
            not math.isfinite(self.sigma2) or self.sigma2 <= 0.0
        ):
            raise ValueError("sigma2 must be a finite positive value or None")

        self._feature_dim = len(_featurize(StrategySpec()))
        self._X: List[np.ndarray] = []
        self._y: List[float] = []

    def observe(self, spec: StrategySpec, score: float) -> None:
        """Record the score of an evaluated strategy (lower is better).

        Args:
            spec: The evaluated strategy.
            score: Its score.

        Raises:
            ValueError: If ``score`` is not finite.
        """
        score_value = float(score)
        if not math.isfinite(score_value):
            raise ValueError("score must be finite")

        self._X.append(_featurize(spec))
        self._y.append(score_value)

    def _posterior(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the posterior mean and precision matrix of the weights.

        The precision is ``alpha * I + X.T @ X / sigma2`` and the mean solves
        ``precision @ mean = X.T @ y / sigma2``; with no observations they are
        the prior's, zero and ``alpha * I``.
        """
        if not self._X:
            mean = np.zeros(self._feature_dim)
            precision = np.eye(self._feature_dim) * self.alpha
            return mean, precision

        X = np.vstack(self._X)
        y = np.asarray(self._y)
        sigma2 = self._noise_variance()
        # Ridge posterior precision (inverse covariance).
        precision = self.alpha * np.eye(X.shape[1]) + (X.T @ X) / sigma2
        rhs = (X.T @ y) / sigma2
        mean = np.linalg.solve(precision, rhs)
        return mean, precision

    def _noise_variance(self) -> float:
        """Return the observation-noise variance the posterior assumes."""
        if self.sigma2 is not None:
            return float(self.sigma2)
        if len(self._y) < 2:
            return _PRIOR_NOISE_VARIANCE
        return max(float(np.var(self._y, ddof=1)), _MIN_NOISE_VARIANCE)

    def propose(
        self, candidates: Sequence[StrategySpec], rng: np.random.Generator
    ) -> StrategySpec:
        """Return the candidate a posterior sample of the weights scores lowest.

        Args:
            candidates: Strategies to choose from.
            rng: Generator the weights are sampled with.

        Returns:
            One of ``candidates``; ties go to the first.

        Raises:
            ValueError: If ``candidates`` is empty.
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
