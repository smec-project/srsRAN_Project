"""Configuration management for Tutti Controller."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ControllerConfig:
    """Configuration settings for the Tutti Controller.
    
    This class holds all configuration parameters needed to initialize
    and run the Tutti Controller system.
    """
    
    # Network ports and addresses
    app_port: int = 5557  # Port to receive application messages
    ran_metrics_ip: str = "127.0.0.1"  # RAN metrics server IP
    ran_metrics_port: int = 5556  # Port to receive RAN metrics
    ran_control_ip: str = "127.0.0.1"  # RAN control server IP
    ran_control_port: int = 5555  # Port to send priority updates
    
    # System parameters
    history_window: int = 100  # Number of slots to keep in PRB history
    log_file_path: str = "controller.txt"  # Path to log file
    
    # Priority calculation constants
    bytes_per_prb: int = 80  # Bytes per Physical Resource Block
    tti_duration_ms: float = 0.5  # Transmission Time Interval in milliseconds
    default_priority_offset: float = 1.0  # Initial priority offset
    max_latency_threshold_ms: int = 3000  # Max latency requirement for priority adjustment
    ms_to_seconds: float = 0.001  # Conversion factor from milliseconds to seconds


class DefaultValues:
    """Default configuration values used throughout the system."""
    
    SOCKET_BUFFER_SIZE: int = 1024  # UDP socket receive buffer size
    MESSAGE_SIZE: int = 20  # Expected UDP message size for app messages
    METRICS_MESSAGE_SIZE: int = 16  # Expected UDP message size for RAN metrics
    MS_TO_SECONDS: float = 0.001  # Conversion factor from milliseconds to seconds