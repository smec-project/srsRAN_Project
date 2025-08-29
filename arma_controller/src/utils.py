"""Utility functions and classes for Tutti Controller."""

import time
from typing import Optional, TextIO


class Logger:
    """Simple logger for writing controller events to file."""
    
    def __init__(self, log_file_path: str):
        """Initialize the logger with a file path.
        
        Args:
            log_file_path: Path to the log file.
        """
        self.log_file: Optional[TextIO] = None
        self.log_file_path = log_file_path
        self._open_log_file()
    
    def _open_log_file(self) -> None:
        """Open the log file for writing."""
        try:
            self.log_file = open(self.log_file_path, 'w')
        except Exception as e:
            print(f"Failed to open log file {self.log_file_path}: {e}")
    
    def write(self, message: str) -> None:
        """Write a message to the log file.
        
        Args:
            message: The message to write.
        """
        if self.log_file:
            self.log_file.write(message)
    
    def flush(self) -> None:
        """Flush the log file buffer."""
        if self.log_file:
            self.log_file.flush()
    
    def close(self) -> None:
        """Close the log file."""
        if self.log_file:
            self.log_file.close()
            self.log_file = None
    
    def __del__(self):
        """Ensure log file is closed when logger is destroyed."""
        self.close()


def get_current_time_ms() -> float:
    """Get current time in milliseconds.
    
    Returns:
        Current time in milliseconds since epoch.
    """
    return time.time() * 1000.0


def rnti_to_hex_string(rnti: int) -> str:
    """Convert RNTI integer to 4-digit hex string.
    
    Args:
        rnti: The RNTI value as integer.
        
    Returns:
        RNTI as 4-digit hex string (e.g., "0001").
    """
    return f"{rnti:04x}"


def calculate_prbs_needed(request_size: int, bytes_per_prb: int) -> int:
    """Calculate number of PRBs needed for a given request size.
    
    Args:
        request_size: Size of the request in bytes.
        bytes_per_prb: Number of bytes per Physical Resource Block.
        
    Returns:
        Number of PRBs needed (rounded up).
    """
    return (request_size + bytes_per_prb - 1) // bytes_per_prb