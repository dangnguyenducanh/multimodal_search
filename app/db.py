"""Read-only SQLite access (backend.md §9.3, §9.9)."""
import gzip
import shutil
import sqlite3
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path

from app.config import DB_PATH
from app.errors import ApiError

_unpack_lock = threading.Lock()


def db_file() -> Path:
    """The DB path; on Vercel only `<db>.gz` ships (100 MB per-file upload cap), unpacked once per cold start."""
    if DB_PATH.is_file():
        return DB_PATH
    gz = DB_PATH.with_name(DB_PATH.name + ".gz")
    if not gz.is_file():
        raise ApiError(503, "NOT_READY", "catalog database missing; run scripts/p9_build_db.py")
    tmp = Path(tempfile.gettempdir()) / DB_PATH.name
    with _unpack_lock:
        if not tmp.is_file():
            part = tmp.with_suffix(".part")
            with gzip.open(gz) as src, open(part, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            part.rename(tmp)
    return tmp


def get_db() -> Iterator[sqlite3.Connection]:
    """FastAPI dependency: one read-only connection per request (opening an immutable DB costs microseconds)."""
    con = sqlite3.connect(f"file:{db_file()}?mode=ro&immutable=1", uri=True, check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        yield con
    finally:
        con.close()


def has_table(db: sqlite3.Connection, name: str) -> bool:
    """Model tables (pair_scores, recs) exist only after scripts/p9_add_models.py has run."""
    return db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def product_by_id(db: sqlite3.Connection, product_id: str) -> sqlite3.Row:
    row = db.execute("SELECT * FROM products WHERE product_id = ?", (product_id,)).fetchone()
    if row is None:
        raise ApiError(404, "PRODUCT_NOT_FOUND", f"no product {product_id}")
    return row
