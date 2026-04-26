# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging

LOGGER_NAME = "qbalance"


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """Return logger for the provided inputs.

    Args:
        name (default: LOGGER_NAME): Name/identifier for a circuit, dataset, or lookup record.

    Returns:
        logging.Logger with the computed result.

    Raises:
        None.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
