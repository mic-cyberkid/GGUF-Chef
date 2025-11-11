# memory_manager.py
import os
import pickle
from typing import List, Dict
from threading import Lock
import logging
from datetime import datetime
from tech_support_logger import TechSupportLogger

memory_logger = TechSupportLogger(
    log_file_name="memory_manager.log",
    log_dir="data/logs",
    level=logging.INFO,
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_output=False
).get_logger()

class MemoryManager:
    def __init__(self,
                 max_tokens=2048,
                 memory_limit=20,  # Limit to recent messages only
                 memory_path="memory.pkl"):

        self.max_tokens = max_tokens
        self.memory_limit = memory_limit
        self.recent_history: List[Dict] = []

        self.memory_path = memory_path
        self.lock = Lock()
        self.last_save = datetime.now()

        # Load existing data
        self._load_memory()
        memory_logger.info(f"MemoryManager initialized. Memory path: {self.memory_path}")

    def _safe_save(self):
        """Thread-safe full save operation"""
        try:
            with self.lock:
                memory_copy = self.recent_history.copy()

            # Save memory data
            temp_path = self.memory_path + ".tmp"
            with open(temp_path, "wb") as f:
                pickle.dump(memory_copy, f)
            os.replace(temp_path, self.memory_path)

            self.last_save = datetime.now()
            memory_logger.info("Memory state saved successfully.")

        except Exception as e:
            memory_logger.error(f"Failed to perform save: {e}", exc_info=True)

    def _load_memory(self):
        """Load memory from file"""
        memory_logger.info("Attempting to load existing memory data.")
        try:
            if os.path.exists(self.memory_path):
                with open(self.memory_path, "rb") as f:
                    with self.lock:
                        self.recent_history = pickle.load(f)
                memory_logger.info(f"Memory loaded from {self.memory_path}. Total messages: {len(self.recent_history)}")
            else:
                memory_logger.info(f"No existing memory data found at {self.memory_path}.")

        except Exception as e:
            memory_logger.error(f"Memory load failed: {e}", exc_info=True)

    def add_message(self, role: str, content: str):
        """Add message and trim if needed"""
        try:
            with self.lock:
                message = {"role": role, "content": content[:self.max_tokens * 4]}  # Truncate long messages
                self.recent_history.append(message)
                self._trim_history()

            # Save periodically
            if (datetime.now() - self.last_save).total_seconds() > 60:
                self._safe_save()
                memory_logger.debug("Memory saved (time-based).")

        except Exception as e:
            memory_logger.error(f"Failed to add message (role: {role}): {e}", exc_info=True)
            raise

    def _trim_history(self):
        """Trim recent history based on limit"""
        while len(self.recent_history) > self.memory_limit:
            removed_msg = self.recent_history.pop(0)
            memory_logger.debug(f"Trimmed history. Removed message (role: {removed_msg['role']}). New length: {len(self.recent_history)}")

    def get_context(self, query: str) -> List[Dict]:
        """Return recent history as context"""
        try:
            with self.lock:
                context = self.recent_history[-10:]  # Last 10 messages for context
            memory_logger.debug(f"Context built. Recent history count: {len(context)}")
            return context
        except Exception as e:
            memory_logger.error(f"Context build failed: {e}", exc_info=True)
            return []

    def __del__(self):
        """Cleanup on destruction"""
        memory_logger.info("MemoryManager instance is being deleted. Attempting final save.")
        self._safe_save()
