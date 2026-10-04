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


class _FallbackHandler(logging.StreamHandler):
    """qbalance's default output, active only while the host has no logging.

    Whether the application configured logging is decided per record, not
    once at import: applications usually import their libraries before they
    call ``logging.basicConfig()``.  Decided at import, the handler would stay
    and -- with propagation off to avoid printing every record twice -- keep
    qbalance's records away from the handlers the application installed
    afterwards.  Records always propagate; this handler only prints the ones
    that would otherwise reach no handler at all.
    """

    def handle(self, record: logging.LogRecord) -> bool:
        """Emit ``record`` unless the root logger has handlers of its own."""
        if logging.root.handlers:
            return False
        return bool(super().handle(record))


def _configure_package_logger() -> None:
    """Install qbalance's default handler on the package logger, at most once.

    Handlers belong to the application, so the default handler prints a
    record only while the host has configured no logging of its own (see
    :class:`_FallbackHandler`); once it has, qbalance stays silent and lets
    the host format and route its records, whichever was set up first.

    The handler goes on the ``qbalance`` package logger rather than on every
    module logger.  Attaching a handler per module while records also propagate
    to the root logger emits every message twice as soon as anything calls
    ``logging.basicConfig()``.

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

    handler = _FallbackHandler()
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
    package_logger.addHandler(handler)


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """Return the logger ``name``, installing qbalance's default handler first.

    See :func:`_configure_package_logger` for when a handler is installed.

    Args:
        name (default: LOGGER_NAME): Logger name; module loggers pass
            ``__name__``, which falls under the ``qbalance`` package logger.
    """
    _configure_package_logger()
    return logging.getLogger(name)
