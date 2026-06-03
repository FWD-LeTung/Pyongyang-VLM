"""Logging configuration shared by backend modules."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "system.log"
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s"


def setup_logger(name: str) -> logging.Logger:
    """Return a logger that writes to stdout and ``logs/system.log``.

    Handlers are installed once per logger name to avoid duplicate lines when
    modules are reloaded by tests or notebook runtimes.
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(LOG_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
