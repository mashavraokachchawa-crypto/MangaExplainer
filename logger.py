"""Centralised, low-footprint logging setup.

Rotating file handler keeps the log disk-bounded (default 1 MB x 3) so logs
cannot grow unbounded on a low-RAM machine.
"""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_SETUP_DONE = set()


def setup_logging(cfg, logger_name="mangaexplainer"):
    if logger_name in _SETUP_DONE:
        return logging.getLogger(logger_name)
    log_dir = Path(cfg.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    file_level = getattr(logging, str(cfg.logging.level).upper(), logging.INFO)
    console_level = getattr(
        logging, str(cfg.logging.console_level).upper(), logging.WARNING
    )
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s %(message)s"
    )
    file_handler = RotatingFileHandler(
        log_dir / "mangaexplainer.log",
        maxBytes=int(cfg.logging.max_bytes),
        backupCount=int(cfg.logging.backup_count),
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(file_level)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(console_level)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    _SETUP_DONE.add(logger_name)
    return logger