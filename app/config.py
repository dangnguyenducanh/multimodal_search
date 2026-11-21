"""Settings from env vars (backend.md §9.2). Only what the deployed demo uses today."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.environ.get("MMS_DB_PATH", ROOT / "runs/p9/catalog.sqlite"))
RETRIEVE_K = int(os.environ.get("MMS_RETRIEVE_K", "100"))
WEB_DIST = ROOT / "web/dist"
