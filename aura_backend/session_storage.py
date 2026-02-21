"""
Session Storage Manager
In-memory storage for session data with optional persistence.
"""

import copy
import logging
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SessionStorage:
    """Manages session data in memory with LRU eviction."""

    def __init__(self, max_sessions: int = 100):
        self.max_sessions = max_sessions
        self.sessions: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        logger.info("Session storage initialized (max: %s)", max_sessions)

    def store_session(self, session_id: str, session_data: Dict[str, Any]) -> bool:
        """Store a session in memory."""
        try:
            session_copy = copy.deepcopy(session_data)
            if "stored_at" not in session_copy:
                session_copy["stored_at"] = datetime.utcnow().isoformat() + "Z"

            self.sessions[session_id] = session_copy
            self.sessions.move_to_end(session_id)

            if len(self.sessions) > self.max_sessions:
                oldest = next(iter(self.sessions))
                self.sessions.pop(oldest)
                logger.info("Evicted oldest session: %s", oldest)

            logger.info("Session stored: %s", session_id)
            return True
        except Exception as exc:
            logger.error("Error storing session: %s", str(exc))
            return False

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a session by ID."""
        try:
            session = self.sessions.get(session_id)
            if session is None:
                logger.warning("Session not found: %s", session_id)
                return None

            self.sessions.move_to_end(session_id)
            logger.info("Session retrieved: %s", session_id)
            return copy.deepcopy(session)
        except Exception as exc:
            logger.error("Error retrieving session: %s", str(exc))
            return None

    def list_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent sessions (most recent first), summarized."""
        try:
            safe_limit = max(1, int(limit))
            results: List[Dict[str, Any]] = []
            for session_id in reversed(list(self.sessions.keys())[-safe_limit:]):
                session = self.sessions[session_id]
                results.append(
                    {
                        "session_id": session_id,
                        "hospital_id": session.get("hospital_id"),
                        "verdict": session.get("verdict"),
                        "timestamp": session.get("timestamp"),
                        "anomaly_score": session.get("anomaly_analysis", {}).get("anomaly_score"),
                    }
                )
            return results
        except Exception as exc:
            logger.error("Error listing sessions: %s", str(exc))
            return []

    def list_full_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent full session objects (most recent first)."""
        try:
            safe_limit = max(1, int(limit))
            ids = list(self.sessions.keys())[-safe_limit:]
            return [copy.deepcopy(self.sessions[sid]) for sid in reversed(ids)]
        except Exception as exc:
            logger.error("Error listing full sessions: %s", str(exc))
            return []

    def all_sessions(self) -> List[Dict[str, Any]]:
        """Return all sessions in insertion order."""
        try:
            return [copy.deepcopy(item) for item in self.sessions.values()]
        except Exception as exc:
            logger.error("Error retrieving all sessions: %s", str(exc))
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get session storage statistics."""
        try:
            total = len(self.sessions)
            verdicts: Dict[str, int] = {}
            for session in self.sessions.values():
                verdict = str(session.get("verdict", "UNKNOWN"))
                verdicts[verdict] = verdicts.get(verdict, 0) + 1

            return {
                "total_sessions": total,
                "max_sessions": self.max_sessions,
                "verdicts": verdicts,
                "utilization": round(total / self.max_sessions, 2) if self.max_sessions > 0 else 0.0,
            }
        except Exception as exc:
            logger.error("Error getting statistics: %s", str(exc))
            return {
                "total_sessions": 0,
                "max_sessions": self.max_sessions,
                "verdicts": {},
                "utilization": 0.0,
            }

    def clear_all(self) -> None:
        """Clear all sessions."""
        count = len(self.sessions)
        self.sessions.clear()
        logger.info("Cleared %s sessions", count)

    def delete_session(self, session_id: str) -> bool:
        """Delete a specific session."""
        try:
            if session_id in self.sessions:
                self.sessions.pop(session_id)
                logger.info("Deleted session: %s", session_id)
                return True
            logger.warning("Session not found for deletion: %s", session_id)
            return False
        except Exception as exc:
            logger.error("Error deleting session: %s", str(exc))
            return False