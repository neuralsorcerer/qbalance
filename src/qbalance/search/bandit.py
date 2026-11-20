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
from qbalance.utils import finite_float_or_none


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
        alpha = finite_float_or_none(self.alpha)
        if alpha is None or alpha <= 0.0:
            raise ValueError("alpha must be a finite positive value")
        self.alpha = alpha
        if self.sigma2 is not None:
            sigma2 = finite_float_or_none(self.sigma2)
            if sigma2 is None or sigma2 <= 0.0:
                raise ValueError("sigma2 must be a finite positive value or None")
            self.sigma2 = sigma2

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
        score_value = finite_float_or_none(score)
        if score_value is None:
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
        inverse_std = self._inverse_noise_std()
        weighted_X = X * inverse_std
        precision = self.alpha * np.eye(X.shape[1]) + weighted_X.T @ weighted_X

        # Normalize the observations for the solve. Even a finite posterior
        # mean can have an unrepresentable unscaled X.T @ y.
        score_scale = float(np.max(np.abs(y))) or 1.0
        rhs = X.T @ (y / score_scale)
        if inverse_std > 1.0:
            # Factor the noise precision out of both sides of the solve.
            mean = np.linalg.solve(precision / inverse_std / inverse_std, rhs)
            mean *= score_scale
        else:
            mean = np.linalg.solve(precision, rhs)
            mean *= (score_scale * inverse_std) * inverse_std
        return mean, precision

    def _inverse_noise_std(self) -> float:
        """Compute reciprocal noise dispersion without squaring large scores."""
        if self.sigma2 is not None:
            return 1.0 / math.sqrt(self.sigma2)
        if len(self._y) < 2:
            return 1.0 / math.sqrt(_PRIOR_NOISE_VARIANCE)
        scores = np.asarray(self._y)
        scale = float(np.max(np.abs(scores))) or 1.0
        relative_std = float(np.std(scores / scale, ddof=1))
        max_precision_sqrt = 1.0 / math.sqrt(_MIN_NOISE_VARIANCE)
        if relative_std == 0.0:
            return max_precision_sqrt
        return min((1.0 / relative_std) / scale, max_precision_sqrt)

    def _noise_variance(self) -> float:
        """Return the observation-noise variance the posterior assumes."""
        if self.sigma2 is not None:
            return float(self.sigma2)
        std = 1.0 / self._inverse_noise_std()
        return std * std

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
        scores = features @ (w / (float(np.max(np.abs(w))) or 1.0))
        best_idx = int(np.argmin(scores))
        return candidates[best_idx]
