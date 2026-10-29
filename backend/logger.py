import os
import sys
import time
import json
import logging
from logging.handlers import RotatingFileHandler
from collections import deque
from datetime import datetime
from typing import List, Dict, Optional, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "app.log")

class SystemLogger:
    def __init__(self, max_buffer: int = 1000):
        self.buffer = deque(maxlen=max_buffer)
        self.file_logger = logging.getLogger("rvk_safety_system")
        self.file_logger.setLevel(logging.DEBUG)

        if not self.file_logger.handlers:
            handler = RotatingFileHandler(
                LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
            )
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "category": "%(name)s", "message": %(message)s}',
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            self.file_logger.addHandler(handler)

        # Log initial startup entry
        self.info("SYSTEM", "Система РВК.Safety CV запущена", {"version": "3.2 SOTA", "mode": "production"})

    def _add_entry(self, level: str, category: str, message: str, meta: Optional[Dict[str, Any]] = None):
        entry = {
            "id": f"log_{int(time.time() * 1000)}_{len(self.buffer)}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "time_short": datetime.now().strftime("%H:%M:%S"),
            "level": level.upper(),
            "category": category.upper(),
            "message": message,
            "meta": meta or {}
        }
        self.buffer.append(entry)

        try:
            log_msg = json.dumps({"text": message, "meta": meta or {}}, ensure_ascii=False)
            if level.upper() == "DEBUG":
                self.file_logger.debug(log_msg)
            elif level.upper() in ("WARNING", "WARN"):
                self.file_logger.warning(log_msg)
            elif level.upper() == "ERROR":
                self.file_logger.error(log_msg)
            else:
                self.file_logger.info(log_msg)
        except Exception:
            pass

    def debug(self, category: str, message: str, meta: Optional[Dict[str, Any]] = None):
        self._add_entry("DEBUG", category, message, meta)

    def info(self, category: str, message: str, meta: Optional[Dict[str, Any]] = None):
        self._add_entry("INFO", category, message, meta)

    def warning(self, category: str, message: str, meta: Optional[Dict[str, Any]] = None):
        self._add_entry("WARNING", category, message, meta)

    def error(self, category: str, message: str, meta: Optional[Dict[str, Any]] = None):
        self._add_entry("ERROR", category, message, meta)

    def audit(self, category: str, message: str, user: str = "admin", meta: Optional[Dict[str, Any]] = None):
        m = meta or {}
        m["user"] = user
        self._add_entry("AUDIT", category, message, m)

    def get_logs(self, limit: int = 150, level: Optional[str] = None, category: Optional[str] = None, search: Optional[str] = None) -> List[Dict]:
        entries = list(self.buffer)
        entries.reverse()

        if level and level.upper() != "ALL":
            entries = [e for e in entries if e["level"] == level.upper()]
        if category and category.upper() != "ALL":
            entries = [e for e in entries if e["category"] == category.upper()]
        if search:
            q = search.lower()
            entries = [e for e in entries if q in e["message"].lower() or q in json.dumps(e["meta"]).lower()]

        return entries[:limit]

    def clear(self):
        self.buffer.clear()
        self.info("AUDIT", "Журнал событий очищен администратором")
        return {"status": "ok", "message": "Буфер логов очищен"}

sys_logger = SystemLogger()
