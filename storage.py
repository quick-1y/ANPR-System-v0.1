import os
import sqlite3
from datetime import datetime, timezone
from typing import List, Optional, Sequence

import aiosqlite

from logging_manager import get_logger


class EventDatabase:
    """SQLite-хранилище для последних распознанных номеров."""

    def __init__(self, db_path: str = "data/events.db") -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
        self.logger = get_logger(__name__)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    plate TEXT NOT NULL,
                    confidence REAL,
                    source TEXT
                )
                """
            )
            conn.commit()

    def insert_event(
        self,
        channel: str,
        plate: str,
        confidence: float = 0.0,
        source: str = "",
        timestamp: Optional[str] = None,
    ) -> int:
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO events (timestamp, channel, plate, confidence, source) VALUES (?, ?, ?, ?, ?)",
                (ts, channel, plate, confidence, source),
            )
            conn.commit()
            self.logger.info(
                "Event saved: %s (%s, conf=%.2f, src=%s)", plate, channel, confidence or 0.0, source
            )
            return cursor.lastrowid

    def fetch_recent(self, limit: int = 100) -> List[sqlite3.Row]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM events ORDER BY datetime(timestamp) DESC LIMIT ?",
                (limit,),
            )
            return cursor.fetchall()

    def fetch_filtered(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        channel: Optional[str] = None,
        plates: Optional[Sequence[str]] = None,
        limit: int = 100,
    ) -> List[sqlite3.Row]:
        filters = []
        params: List[object] = []

        if start:
            filters.append("datetime(timestamp) >= datetime(?)")
            params.append(start)
        if end:
            filters.append("datetime(timestamp) <= datetime(?)")
            params.append(end)
        if channel:
            filters.append("channel = ?")
            params.append(channel)
        if plates:
            placeholders = ",".join("?" for _ in plates)
            filters.append(f"plate IN ({placeholders})")
            params.extend(list(plates))

        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        query = f"SELECT * FROM events {where_clause} ORDER BY datetime(timestamp) DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, tuple(params))
            return cursor.fetchall()

    def search_by_plate(
        self,
        plate_fragment: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> List[sqlite3.Row]:
        filters = ["plate LIKE ?"]
        params: List[object] = [f"%{plate_fragment}%"]

        if start:
            filters.append("datetime(timestamp) >= datetime(?)")
            params.append(start)
        if end:
            filters.append("datetime(timestamp) <= datetime(?)")
            params.append(end)

        where_clause = f"WHERE {' AND '.join(filters)}"
        query = f"SELECT * FROM events {where_clause} ORDER BY datetime(timestamp) DESC"

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, tuple(params))
            return cursor.fetchall()

    def list_channels(self) -> List[str]:
        with self._connect() as conn:
            cursor = conn.execute("SELECT DISTINCT channel FROM events ORDER BY channel")
            return [row[0] for row in cursor.fetchall()]


class AsyncEventDatabase:
    """Асинхронный доступ к SQLite для фоновых потоков распознавания."""

    def __init__(self, db_path: str = "data/events.db") -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._initialized = False
        self.logger = get_logger(__name__)

    async def _ensure_schema(self) -> None:
        if self._initialized:
            return
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    plate TEXT NOT NULL,
                    confidence REAL,
                    source TEXT
                )
                """
            )
            await conn.commit()
        self._initialized = True

    async def insert_event_async(
        self,
        channel: str,
        plate: str,
        confidence: float = 0.0,
        source: str = "",
        timestamp: Optional[str] = None,
    ) -> int:
        await self._ensure_schema()
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.execute(
                "INSERT INTO events (timestamp, channel, plate, confidence, source) VALUES (?, ?, ?, ?, ?)",
                (ts, channel, plate, confidence, source),
            )
            await conn.commit()
            self.logger.info(
                "[async] Event saved: %s (%s, conf=%.2f, src=%s)",
                plate,
                channel,
                confidence or 0.0,
                source,
            )
            return cursor.lastrowid
