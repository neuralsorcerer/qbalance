# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging

LOGGER_NAME = "qbalance"
_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _configure_package_logger() -> None:
    """Install qbalance's default handler on the package logger, at most once.

    Handlers belong to the application, so one is installed only when the host
    has configured no logging of its own; otherwise qbalance stays silent and
    lets the host format and route its records.

    The handler goes on the ``qbalance`` package logger rather than on every
    module logger, and propagation is stopped alongside it.  Attaching a handler
    per module while records also propagate to the root logger emits every
    message twice as soon as anything calls ``logging.basicConfig()``.

    Args:
        None.

    Returns:
        None. This method updates state or performs side effects only.

    Raises:
        None.
    """
    package_logger = logging.getLogger(LOGGER_NAME)
    if package_logger.handlers:
        return
    if logging.root.handlers:
        # The application owns logging output; do not compete with it.
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
    package_logger.addHandler(handler)
    package_logger.setLevel(logging.INFO)
    package_logger.propagate = False


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """Return logger for the provided inputs.

    Args:
        name (default: LOGGER_NAME): Name/identifier for a circuit, dataset, or lookup record.

    Returns:
        logging.Logger with the computed result.

    Raises:
        None.
    """
    _configure_package_logger()
    return logging.getLogger(name)
