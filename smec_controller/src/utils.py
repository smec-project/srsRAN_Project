"""Utility functions and logging for SMEC Controller."""

import time
from typing import Optional, TextIO


class Logger:
    """Simple logger for the SMEC Controller.
    
    Provides basic logging functionality with optional file output.
    """
    
    def __init__(self, enable_logging: bool = False, log_file_path: str = "controller.txt"):
        """Initialize the logger.
        
        Args:
            enable_logging: Whether to enable file logging.
            log_file_path: Path to the log file.
        """
        self.enable_logging = enable_logging
        self.log_file: Optional[TextIO] = None
        
        if self.enable_logging:
            self.log_file = open(log_file_path, "w")
    
    def log(self, message: str) -> None:
        """Log a message to file if logging is enabled.
        
        Args:
            message: The message to log.
        """
        if self.enable_logging and self.log_file:
            timestamp = time.time()
            self.log_file.write(f"[{timestamp:.6f}] {message}\n")
            self.log_file.flush()
    
    def close(self) -> None:
        """Close the log file."""
        if self.log_file:
            self.log_file.close()
            self.log_file = None

def get_current_timestamp() -> float:
    """Get current timestamp in seconds.
    
    Returns:
        Current timestamp as float.
    """
    return time.time()


def calculate_elapsed_time_ms(start_time: float, end_time: Optional[float] = None) -> float:
    """Calculate elapsed time in milliseconds.
    
    Args:
        start_time: Start timestamp in seconds.
        end_time: End timestamp in seconds. Uses current time if None.
        
    Returns:
        Elapsed time in milliseconds.
    """
    if end_time is None:
        end_time = get_current_timestamp()
    return (end_time - start_time) * 1000.0


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, avoiding division by zero.
    
    Args:
        numerator: The numerator.
        denominator: The denominator.
        default: Default value to return if denominator is zero.
        
    Returns:
        Result of division or default value.
    """
    if abs(denominator) < 1e-10:
        return default
    return numerator / denominator


def normalize_cyclic_slot(
    current_slot: int, 
    previous_slot: int, 
    slot_max: int = 20480
) -> tuple[int, bool]:
    """Normalize cyclic slot numbers to detect wrap-around.
    
    Args:
        current_slot: Current slot number.
        previous_slot: Previous slot number.
        slot_max: Maximum slot value before wrap-around.
        
    Returns:
        Tuple of (normalized_slot, wrapped_around).
    """
    wrapped_around = current_slot < previous_slot
    if wrapped_around:
        normalized_slot = current_slot + slot_max
    else:
        normalized_slot = current_slot
    return normalized_slot, wrapped_around 