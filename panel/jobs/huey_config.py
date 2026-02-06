"""Huey configuration with SQLite backend."""
from huey import SqliteHuey

from config import DATA_DIR

DATA_DIR.mkdir(exist_ok=True)

# SQLite-backed Huey instance
huey = SqliteHuey(filename=str(DATA_DIR / 'huey.db'))
