import logging
import sys
import uuid
from typing import Optional

class RunIDFilter(logging.Filter):
    """Attach a run identifier to every log record."""
    def __init__(self, xid: str) -> None:
        super().__init__()
        self.xid = xid

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - trivial
        record.xid = self.xid
        return True

def setup_logger(verbose: bool = False, xid: Optional[str] = None) -> tuple[logging.Logger, str]:
    """Return a configured logger and the run identifier used."""
    xid = xid or str(uuid.uuid4())
    logger = logging.getLogger("QCH")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter('%(asctime)s - %(xid)s - %(levelname)s - %(message)s')
    handler.setFormatter(fmt)
    handler.addFilter(RunIDFilter(xid))
    logger.handlers.clear()
    logger.addHandler(handler)
    return logger, xid
