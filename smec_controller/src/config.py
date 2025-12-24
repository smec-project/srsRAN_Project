"""Configuration constants and settings for SMEC Controller."""

from dataclasses import dataclass


@dataclass
class ControllerConfig:
    """Configuration settings for the SMEC Controller.

    Attributes:
        slo_ctrl_port: Port to receive SLO control plane messages from users.
        ran_metrics_ip: IP address for RAN metrics connection.
        ran_metrics_port: Port for RAN metrics connection.
        ran_control_ip: IP address for RAN control connection.
        ran_control_port: Port for RAN control connection.
        enable_logging: Whether to enable file logging.
        window_size: Size of the sliding window for analysis.
        min_ddl: Minimum deadline value in milliseconds.
        priority_update_interval: Interval for priority updates in seconds.
        slot_duration_ms: Duration of each slot in milliseconds.
        collect_logs_only: If True, only collect logs without priority adjustment.
    """
    slo_ctrl_port: int = 5557
    ran_metrics_ip: str = "127.0.0.1"
    ran_metrics_port: int = 5556
    ran_control_ip: str = "127.0.0.1"
    ran_control_port: int = 5555
    enable_logging: bool = False
    window_size: int = 5
    min_ddl: float = 0.1
    priority_update_interval: float = 0.002
    slot_duration_ms: float = 0.5
    collect_logs_only: bool = False


# Event type constants
class EventTypes:
    """Constants for different event types."""
    SR = 0
    BSR = 1
    PRB = 2
    
    @classmethod
    def get_name_mapping(cls) -> dict:
        """Get mapping from values to names."""
        return {
            cls.SR: "SR",
            cls.BSR: "BSR", 
            cls.PRB: "PRB"
        }


# Network constants
class NetworkConstants:
    """Constants for network operations."""
    SLOT_MAX = 20480
    RECV_BUFFER_SIZE = 1024
    SOCKET_LISTEN_BACKLOG = 5


# Message types
class MessageTypes:
    """Constants for message types."""
    SLO_MESSAGE = 0


# Default file paths
class DefaultPaths:
    """Default file paths for data files."""
    LOG_FILE = "controller.log" 