"""Console and rotating-file logging configuration."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Final

from crawlforge.config import LoggingConfig

_CONSOLE_HANDLER: Final = "crawlforge.console"
_FILE_HANDLER: Final = "crawlforge.file"
_FORMAT: Final = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(config: LoggingConfig) -> None:
    """Configure timestamped console and optional rotating-file logging."""
    root = logging.getLogger()
    for handler in tuple(root.handlers):
        if handler.get_name() in {_CONSOLE_HANDLER, _FILE_HANDLER}:
            root.removeHandler(handler)
            handler.close()

    level = getattr(logging, config.level)
    formatter = logging.Formatter(_FORMAT)
    console = logging.StreamHandler()
    console.set_name(_CONSOLE_HANDLER)
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    if config.file is not None:
        path = Path(config.file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            path,
            maxBytes=config.max_bytes,
            backupCount=config.backup_count,
            encoding="utf-8",
        )
        file_handler.set_name(_FILE_HANDLER)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    root.setLevel(level)
