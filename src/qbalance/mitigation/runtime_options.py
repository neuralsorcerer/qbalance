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
    """Build runtime estimator options from the provided configuration parameters.

    Args:
        resilience_level (default: 1): Resilience level value consumed by this routine.
        enable_gate_twirling (default: None): Enable gate twirling value consumed by this routine.
        enable_measurement_mitigation (default: None): Enable measurement mitigation value consumed by this routine.
        enable_zne (default: None): Enable zne value consumed by this routine.
        layer_noise_model (default: None): Layer noise model value consumed by this routine.

    Returns:
        Dict[str, Any] with the computed result.

    Raises:
        None.
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
