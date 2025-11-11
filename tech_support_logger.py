# tech_support_logger.py

import logging
import os
from logging.handlers import RotatingFileHandler

class TechSupportLogger:
    """
    A custom logging class for the Physics Chatbot Middleware.
    It logs messages to a file, supports log rotation, and allows for
    different logging levels. This log file can be sent to tech support
    for troubleshooting.
    """

    def __init__(self,
                 log_file_name: str = "middleware.log",
                 log_dir: str = "logs",
                 level: int = logging.INFO,
                 max_bytes: int = 10 * 1024 * 1024,  # 10 MB
                 backup_count: int = 5,
                 console_output: bool = True):
        """
        Initializes the custom logger.

        Args:
            log_file_name (str): The name of the log file (e.g., "middleware.log").
            log_dir (str): The directory where log files will be stored.
                           Defaults to "logs" relative to the script's location.
            level (int): The minimum logging level to capture (e.g., logging.INFO, logging.DEBUG).
            max_bytes (int): The maximum size of a log file in bytes before rotation occurs.
                             Defaults to 10 MB.
            backup_count (int): The number of backup log files to keep.
                                 Defaults to 5 (e.g., middleware.log.1, middleware.log.2, etc.).
            console_output (bool): If True, logs will also be printed to the console.
        """
        self.log_dir = log_dir
        self.log_file_path = os.path.join(self.log_dir, log_file_name)
        self.level = level
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.console_output = console_output

        # Ensure the log directory exists
        os.makedirs(self.log_dir, exist_ok=True)

        self._logger = logging.getLogger("PhysicsChatbotMiddleware")
        self._logger.setLevel(self.level)

        # Prevent adding multiple handlers if the logger is already configured
        # This is important if TechSupportLogger is instantiated multiple times
        # or if other modules also configure logging.
        if not self._logger.handlers:
            self._setup_handlers()
        else:
            # If handlers already exist, ensure they are removed to prevent duplicate logs
            # This can happen if logging.basicConfig was called elsewhere before this.
            # A more robust approach might be to check if the *correct* handlers are present.
            # For this scenario, we'll clear existing handlers if present to ensure our setup.
            for handler in list(self._logger.handlers):
                self._logger.removeHandler(handler)
            self._setup_handlers()


    def _setup_handlers(self):
        """
        Sets up the file handler (with rotation) and optionally the console handler.
        """
        # Formatter for log messages
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s'
        )

        # Rotating File Handler
        # This handler writes logs to a file and rotates them when the file
        # reaches a certain size. It keeps a specified number of backup files.
        file_handler = RotatingFileHandler(
            self.log_file_path,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)

        # Console Handler (optional)
        if self.console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)

    def debug(self, message: str):
        """Logs a message with DEBUG level."""
        self._logger.debug(message)

    def info(self, message: str):
        """Logs a message with INFO level."""
        self._logger.info(message)

    def warning(self, message: str):
        """Logs a message with WARNING level."""
        self._logger.warning(message)

    def error(self, message: str, exc_info: bool = False):
        """
        Logs a message with ERROR level.
        Args:
            message (str): The error message.
            exc_info (bool): If True, exception information (traceback) is added to the log record.
        """
        self._logger.error(message, exc_info=exc_info)

    def critical(self, message: str, exc_info: bool = False):
        """
        Logs a message with CRITICAL level.
        Args:
            message (str): The critical message.
            exc_info (bool): If True, exception information (traceback) is added to the log record.
        """
        self._logger.critical(message, exc_info=exc_info)

    def get_logger(self) -> logging.Logger:
        """Returns the underlying logging.Logger instance."""
        return self._logger

# Example Usage (for testing the logger itself)
if __name__ == "__main__":
    # Initialize the logger
    # Logs will go to 'logs/app.log' and also print to console
    app_logger = TechSupportLogger(
        log_file_name="app_test.log",
        log_dir="test_logs", # Using a subdirectory for testing
        level=logging.DEBUG, # Set to DEBUG to capture all messages
        max_bytes=1 * 1024 * 1024, # 1 MB for testing rotation
        backup_count=3,
        console_output=True
    )

    # Get the logger instance
    logger = app_logger.get_logger()

    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message!")

    try:
        1 / 0
    except ZeroDivisionError:
        logger.error("An error occurred during division.", exc_info=True)

    logger.info("Generating some more logs to test rotation...")
    for i in range(5000): # Write enough lines to trigger rotation if max_bytes is small
        logger.info(f"Log line {i}: This is a test message to fill up the log file. " * 10)

    logger.info("Log generation complete. Check 'test_logs' directory for files.")
