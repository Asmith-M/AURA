import logging
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = [
    ("tx_id", "TEXT"),
    ("hospital_id", "TEXT"),
    ("update_hash", "TEXT"),
    ("verdict", "TEXT"),
    ("evidence_hash", "TEXT"),
    ("anomaly_score", "REAL"),
    ("timestamp", "TEXT"),
]


class LedgerManager:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.active_db_path = str(self.db_path)
        self.conn: Optional[sqlite3.Connection] = None
        self.last_error: Optional[str] = None
        self.using_memory_fallback = False

    def _connect(self, db_target: Optional[str] = None) -> sqlite3.Connection:
        target = db_target or self.active_db_path
        conn = sqlite3.connect(target, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize_with_connection(self, conn: sqlite3.Connection) -> None:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                tx_id TEXT PRIMARY KEY,
                hospital_id TEXT NOT NULL,
                update_hash TEXT NOT NULL,
                verdict TEXT NOT NULL,
                evidence_hash TEXT NOT NULL,
                anomaly_score REAL NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        self._migrate_schema_if_needed(cursor)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp DESC)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_transactions_hospital ON transactions(hospital_id)"
        )
        conn.commit()

    def initialize_database(self) -> bool:
        try:
            self.active_db_path = str(self.db_path)
            self.using_memory_fallback = False
            self.conn = self._connect(self.active_db_path)
            self._initialize_with_connection(self.conn)
            self.last_error = None
            return True
        except Exception as exc:
            if self.conn is not None:
                try:
                    self.conn.close()
                except Exception:
                    pass
            self.conn = None

            primary_error = str(exc)
            fallback_candidates = []
            local_recovery = str(self.db_path.with_name(f"{self.db_path.stem}_recovery.db"))
            temp_recovery = str(Path(tempfile.gettempdir()) / f"{self.db_path.stem}_recovery.db")
            for candidate in (local_recovery, temp_recovery):
                if candidate not in fallback_candidates:
                    fallback_candidates.append(candidate)

            fallback_errors: List[str] = []
            for fallback_path in fallback_candidates:
                try:
                    logger.warning(
                        "Primary ledger database unavailable (%s). Falling back to %s",
                        primary_error,
                        fallback_path,
                    )
                    self.active_db_path = fallback_path
                    self.using_memory_fallback = False
                    self.conn = self._connect(self.active_db_path)
                    self._initialize_with_connection(self.conn)
                    self.last_error = f"Primary DB failed ({primary_error}); using fallback {fallback_path}"
                    return True
                except Exception as fallback_exc:
                    fallback_errors.append(f"{fallback_path}: {fallback_exc}")
                    if self.conn is not None:
                        try:
                            self.conn.close()
                        except Exception:
                            pass
                    self.conn = None

            try:
                logger.warning(
                    "File-based ledger fallbacks failed (%s). Falling back to in-memory ledger.",
                    "; ".join(fallback_errors),
                )
                self.active_db_path = ":memory:"
                self.using_memory_fallback = True
                self.conn = self._connect(self.active_db_path)
                self._initialize_with_connection(self.conn)
                self.last_error = (
                    f"Primary DB failed ({primary_error}); file fallbacks failed ({'; '.join(fallback_errors)}); "
                    "using in-memory ledger"
                )
                return True
            except Exception as mem_exc:
                if self.conn is not None:
                    try:
                        self.conn.close()
                    except Exception:
                        pass
                self.conn = None
                self.active_db_path = str(self.db_path)
                self.using_memory_fallback = False
                self.last_error = (
                    f"Primary DB failed ({primary_error}); file fallbacks failed ({'; '.join(fallback_errors)}); "
                    f"memory fallback failed ({mem_exc})"
                )
                logger.error("Ledger database initialization failed: %s", self.last_error)
                return False

    def _migrate_schema_if_needed(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute("PRAGMA table_info(transactions)")
        existing = {row[1] for row in cursor.fetchall()}
        required = {name for name, _dtype in _REQUIRED_COLUMNS}

        missing = required - existing
        if not missing:
            return

        logger.warning("Ledger schema missing columns %s; running migration", sorted(missing))

        cursor.execute("ALTER TABLE transactions RENAME TO transactions_old")
        cursor.execute(
            """
            CREATE TABLE transactions (
                tx_id TEXT PRIMARY KEY,
                hospital_id TEXT NOT NULL,
                update_hash TEXT NOT NULL,
                verdict TEXT NOT NULL,
                evidence_hash TEXT NOT NULL,
                anomaly_score REAL NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )

        cursor.execute("PRAGMA table_info(transactions_old)")
        old_cols = {row[1] for row in cursor.fetchall()}

        select_parts = []
        for col, _dtype in _REQUIRED_COLUMNS:
            if col in old_cols:
                select_parts.append(col)
            elif col == "evidence_hash":
                select_parts.append("'' AS evidence_hash")
            elif col == "anomaly_score":
                select_parts.append("0.0 AS anomaly_score")
            elif col == "timestamp":
                select_parts.append("datetime('now') AS timestamp")
            else:
                select_parts.append(f"'' AS {col}")

        cursor.execute(
            f"""
            INSERT OR IGNORE INTO transactions ({', '.join(col for col, _ in _REQUIRED_COLUMNS)})
            SELECT {', '.join(select_parts)}
            FROM transactions_old
            """
        )
        cursor.execute("DROP TABLE transactions_old")

    def _ensure_conn(self) -> Optional[sqlite3.Connection]:
        if self.conn is None:
            if not self.initialize_database():
                return None
        return self.conn

    def is_connected(self) -> bool:
        return self._ensure_conn() is not None

    def log_transaction(
        self,
        tx_id: str,
        hospital_id: str,
        update_hash: str,
        verdict: str,
        evidence_hash: str,
        anomaly_score: float,
    ) -> bool:
        def _insert_once() -> bool:
            conn = self._ensure_conn()
            if conn is None:
                return False
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO transactions (
                    tx_id, hospital_id, update_hash, verdict, evidence_hash, anomaly_score, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tx_id,
                    hospital_id,
                    update_hash,
                    verdict,
                    evidence_hash,
                    float(anomaly_score),
                    datetime.utcnow().isoformat() + "Z",
                ),
            )
            conn.commit()
            return True

        try:
            return _insert_once()
        except sqlite3.IntegrityError:
            logger.warning("Duplicate transaction attempted: %s", tx_id)
            return False
        except sqlite3.OperationalError as exc:
            self.last_error = str(exc)
            logger.warning(
                "Ledger insert operational error for %s (%s). Reinitializing ledger and retrying once.",
                tx_id,
                str(exc),
            )
            if self.conn is not None:
                try:
                    self.conn.close()
                except Exception:
                    pass
            self.conn = None

            if not self.initialize_database():
                logger.error("Ledger reinitialization failed after insert error for %s", tx_id)
                return False

            try:
                return _insert_once()
            except sqlite3.IntegrityError:
                logger.warning("Duplicate transaction attempted after retry: %s", tx_id)
                return False
            except Exception as retry_exc:
                logger.error("Ledger insert retry failed for %s: %s", tx_id, str(retry_exc))
                self.last_error = str(retry_exc)
                return False
        except Exception as exc:
            logger.error("Ledger insert failed for %s: %s", tx_id, str(exc))
            self.last_error = str(exc)
            return False

    def get_transaction(self, tx_id: str) -> Optional[Dict[str, Any]]:
        conn = self._ensure_conn()
        if conn is None:
            return None
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM transactions WHERE tx_id = ?", (tx_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        except Exception as exc:
            logger.error("Ledger read failed for %s: %s", tx_id, str(exc))
            self.last_error = str(exc)
            return None

    def get_recent_transactions(self, limit: int = 20) -> List[Dict[str, Any]]:
        conn = self._ensure_conn()
        if conn is None:
            return []
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT * FROM transactions ORDER BY timestamp DESC LIMIT ?",
                (max(1, int(limit)),),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as exc:
            logger.error("Ledger recent transactions query failed: %s", str(exc))
            self.last_error = str(exc)
            return []

    def get_statistics(self) -> Dict[str, Any]:
        default_stats = {
            "total_transactions": 0,
            "verdicts": {},
            "approved": 0,
            "rejected": 0,
            "average_anomaly_score": 0.0,
        }
        conn = self._ensure_conn()
        if conn is None:
            return default_stats
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT COUNT(*) AS total FROM transactions")
            total_transactions = int(cursor.fetchone()["total"])

            cursor.execute("SELECT verdict, COUNT(*) AS count FROM transactions GROUP BY verdict")
            verdicts = {row["verdict"]: int(row["count"]) for row in cursor.fetchall()}

            cursor.execute("SELECT AVG(anomaly_score) AS avg_score FROM transactions")
            avg = cursor.fetchone()["avg_score"]

            approved = verdicts.get("APPROVED", 0)
            rejected = verdicts.get("REJECTED", 0)

            return {
                "total_transactions": total_transactions,
                "verdicts": verdicts,
                "approved": approved,
                "rejected": rejected,
                "average_anomaly_score": float(avg) if avg is not None else 0.0,
            }
        except Exception as exc:
            logger.error("Ledger statistics query failed: %s", str(exc))
            self.last_error = str(exc)
            return default_stats

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None
