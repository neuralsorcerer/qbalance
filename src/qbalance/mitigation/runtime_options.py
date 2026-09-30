# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict, Optional


def build_runtime_estimator_options(
    resilience_level: int = 1,
    enable_gate_twirling: Optional[bool] = None,
    enable_measurement_mitigation: Optional[bool] = None,
    enable_zne: Optional[bool] = None,
    layer_noise_model: Any = None,
) -> Dict[str, Any]:
    """Return an IBM Runtime ``EstimatorV2`` options mapping.

    Options left as ``None`` are omitted.  This is a standalone helper:
    qbalance's own compile and execute path does not call it.

    Args:
        resilience_level (default: 1): ``resilience_level``, cast to int.
        enable_gate_twirling (default: None): ``twirling.enable_gates``.
        enable_measurement_mitigation (default: None):
            ``resilience.measure_mitigation``.
        enable_zne (default: None): ``resilience.zne_mitigation``.
        layer_noise_model (default: None): ``resilience.layer_noise_model``,
            passed through unchanged.

    Returns:
        The nested options dict, for example ``{"resilience_level": 2,
        "twirling": {"enable_gates": True}}``.
    """
    opts: Dict[str, Any] = {"resilience_level": int(resilience_level)}
    if enable_gate_twirling is not None:
        opts.setdefault("twirling", {})["enable_gates"] = bool(enable_gate_twirling)
    if enable_measurement_mitigation is not None:
        opts.setdefault("resilience", {})["measure_mitigation"] = bool(
            enable_measurement_mitigation
        )
    if enable_zne is not None:
        opts.setdefault("resilience", {})["zne_mitigation"] = bool(enable_zne)
    if layer_noise_model is not None:
        opts.setdefault("resilience", {})["layer_noise_model"] = layer_noise_model
    return opts
