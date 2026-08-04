from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class PipelineState:
    """SQLite-backed stage/chunk ledger for restartable execution."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def _init_db(self) -> None:
        with self.connect() as con:
            con.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS stages (
                    stage TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    started_at REAL,
                    finished_at REAL,
                    input_hash TEXT,
                    output_path TEXT,
                    metadata_json TEXT,
                    error TEXT
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    stage TEXT NOT NULL,
                    chunk_id INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    started_at REAL,
                    finished_at REAL,
                    output_path TEXT,
                    metadata_json TEXT,
                    error TEXT,
                    PRIMARY KEY(stage, chunk_id)
                );
                """
            )

    def stage_done(self, stage: str, input_hash: str | None = None) -> bool:
        with self.connect() as con:
            row = con.execute("SELECT * FROM stages WHERE stage=?", (stage,)).fetchone()
        if row is None or row["status"] != "done":
            return False
        return input_hash is None or row["input_hash"] == input_hash

    def start_stage(self, stage: str, input_hash: str | None = None, metadata: Any = None) -> None:
        now = time.time()
        with self.connect() as con:
            con.execute(
                """INSERT INTO stages(stage,status,started_at,input_hash,metadata_json,error)
                   VALUES(?,?,?,?,?,NULL)
                   ON CONFLICT(stage) DO UPDATE SET
                   status=excluded.status, started_at=excluded.started_at,
                   input_hash=excluded.input_hash, metadata_json=excluded.metadata_json,
                   finished_at=NULL, output_path=NULL, error=NULL""",
                (stage, "running", now, input_hash, json.dumps(metadata, default=str)),
            )

    def finish_stage(self, stage: str, output_path: str | None = None, metadata: Any = None) -> None:
        with self.connect() as con:
            con.execute(
                "UPDATE stages SET status='done', finished_at=?, output_path=?, metadata_json=?, error=NULL WHERE stage=?",
                (time.time(), output_path, json.dumps(metadata, default=str), stage),
            )

    def fail_stage(self, stage: str, error: str) -> None:
        with self.connect() as con:
            con.execute(
                "UPDATE stages SET status='failed', finished_at=?, error=? WHERE stage=?",
                (time.time(), error, stage),
            )

    def chunk_done(self, stage: str, chunk_id: int) -> bool:
        with self.connect() as con:
            row = con.execute(
                "SELECT status FROM chunks WHERE stage=? AND chunk_id=?", (stage, chunk_id)
            ).fetchone()
        return bool(row and row["status"] == "done")

    def start_chunk(self, stage: str, chunk_id: int, metadata: Any = None) -> None:
        with self.connect() as con:
            con.execute(
                """INSERT INTO chunks(stage,chunk_id,status,started_at,metadata_json,error)
                   VALUES(?,?,?,?,?,NULL)
                   ON CONFLICT(stage,chunk_id) DO UPDATE SET
                   status='running', started_at=excluded.started_at,
                   finished_at=NULL, output_path=NULL,
                   metadata_json=excluded.metadata_json, error=NULL""",
                (stage, chunk_id, "running", time.time(), json.dumps(metadata, default=str)),
            )

    def finish_chunk(
        self, stage: str, chunk_id: int, output_path: str | None = None, metadata: Any = None
    ) -> None:
        with self.connect() as con:
            con.execute(
                """UPDATE chunks SET status='done', finished_at=?, output_path=?,
                   metadata_json=?, error=NULL WHERE stage=? AND chunk_id=?""",
                (time.time(), output_path, json.dumps(metadata, default=str), stage, chunk_id),
            )

    def fail_chunk(self, stage: str, chunk_id: int, error: str) -> None:
        with self.connect() as con:
            con.execute(
                "UPDATE chunks SET status='failed', finished_at=?, error=? WHERE stage=? AND chunk_id=?",
                (time.time(), error, stage, chunk_id),
            )
