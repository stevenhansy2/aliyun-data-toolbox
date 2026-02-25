"""Central logging setup for converter."""

from __future__ import annotations

import builtins
import inspect
import logging
import os
import re
from typing import Any

_PRINT_PATCHED_FLAG = "_kuavo_print_patched"
_LOG_PRINT_INSTALLED_FLAG = "_kuavo_log_print_installed"
_LEVEL_PREFIX = re.compile(r"^\s*\[(ERROR|ERR|WARN|WARNING|INFO|DEBUG|TIME)\]\s*")


def setup_logging(level: str = "INFO") -> None:
    """Initialize root logging once."""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(_parse_level(level))
        return
    logging.basicConfig(
        level=_parse_level(level),
        format="[%(levelname)s][%(name)s] %(message)s",
    )


def patch_project_print(project_hint: str = "rosbag2hdf5/kuavo") -> None:
    """Route project `print` calls to logging to keep output format unified."""
    if getattr(builtins, _PRINT_PATCHED_FLAG, False):
        return

    original_print = builtins.print
    project_logger = logging.getLogger("converter.stdout")

    def logging_print(*args: Any, **kwargs: Any) -> None:
        caller_file = _caller_file()
        if caller_file and project_hint not in caller_file:
            original_print(*args, **kwargs)
            return

        sep = kwargs.get("sep", " ")
        msg = sep.join(str(arg) for arg in args)
        project_logger.info(msg)

    builtins.print = logging_print
    setattr(builtins, _PRINT_PATCHED_FLAG, True)


def log_print(
    *args: Any,
    sep: str = " ",
    logger: logging.Logger | None = None,
    level: int | None = None,
) -> None:
    """Drop-in print replacement backed by logging with auto level detection."""
    _ensure_default_logging()
    msg = sep.join(str(arg) for arg in args)
    use_logger = logger or logging.getLogger(_caller_module_name())
    use_level = level if level is not None else _infer_level_from_message(msg)
    use_logger.log(use_level, _strip_level_prefix(msg))


def install_global_log_print() -> None:
    """Expose log_print in builtins so legacy modules can call it without per-file imports."""
    if getattr(builtins, _LOG_PRINT_INSTALLED_FLAG, False):
        return
    setattr(builtins, "log_print", log_print)
    setattr(builtins, _LOG_PRINT_INSTALLED_FLAG, True)


def _caller_file() -> str:
    for frame_info in inspect.stack()[2:]:
        filename = frame_info.filename
        if filename and filename != __file__:
            return os.path.normpath(filename)
    return ""


def _parse_level(level: str) -> int:
    normalized = (level or "INFO").upper()
    return getattr(logging, normalized, logging.INFO)


def _ensure_default_logging() -> None:
    if not logging.getLogger().handlers:
        setup_logging("INFO")


def _infer_level_from_message(msg: str) -> int:
    m = _LEVEL_PREFIX.match(msg or "")
    if not m:
        return logging.INFO
    tag = m.group(1)
    if tag in ("ERROR", "ERR"):
        return logging.ERROR
    if tag in ("WARN", "WARNING"):
        return logging.WARNING
    if tag == "DEBUG":
        return logging.DEBUG
    return logging.INFO


def _strip_level_prefix(msg: str) -> str:
    return _LEVEL_PREFIX.sub("", msg or "", count=1)


def _caller_module_name() -> str:
    for frame_info in inspect.stack()[2:]:
        frame = frame_info.frame
        module_name = frame.f_globals.get("__name__", "")
        if not module_name:
            continue
        if module_name != __name__:
            return module_name
    return "converter.stdout"
