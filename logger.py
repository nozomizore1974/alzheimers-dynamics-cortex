"""
logger.py -- one configured logger for the pipeline.

Why not bare ``print``: when stdout is a redirected job-log file it is block-
buffered, and a hard NEST/MPI process exit skips the flush, so buffered lines
are lost.  ``logging.StreamHandler`` writes to stderr and flushes after *every*
record, so each line lands in the log immediately and survives a hard exit.

Self-contained: standard library only (no colorlog).  Modules that take a
``logger=`` callable (``cache``, ``simulator``) are fed this logger's ``.info``
method, so the existing ``logger(msg)`` call sites keep working unchanged.
"""
import logging

_FMT = logging.Formatter(
    "%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name="main", level=logging.INFO, log_file=None):
    """Return a configured logger; idempotent (won't add duplicate handlers).

    Parameters
    ----------
    name : str
        Logger name (use a per-rank name under MPI to keep logs separate).
    level : int
        Logging level (``logging.INFO`` by default).
    log_file : str, optional
        If given, also append records to this file (created lazily).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler()  # -> stderr, flushes per record
        sh.setFormatter(_FMT)
        logger.addHandler(sh)

    if log_file and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8", delay=True)
        fh.setFormatter(_FMT)
        logger.addHandler(fh)

    return logger
