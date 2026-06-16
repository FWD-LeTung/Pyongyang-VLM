"""Logging configuration shared by backend modules."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "system.log"
VISION_PIPELINE_LOG_FILE = LOG_DIR / "vision_pipeline.log"
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s"


def _default_log_file(name: str) -> Path:
    """Pick the module-specific log file for a logger name."""

    if name.startswith("src.vision_pipeline"):
        return VISION_PIPELINE_LOG_FILE
    return LOG_FILE


def setup_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    """Return a logger that writes to stdout and a module log file.

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
    target_log_file = Path(log_file) if log_file is not None else _default_log_file(name)
    file_handler = logging.FileHandler(target_log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
