"""Configuration constants and settings for PMEC Controller."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ControllerConfig:
    """Configuration settings for the PMEC Controller.
    
    Attributes:
        slo_ctrl_port: Port to receive SLO control plane messages from users.
        ran_metrics_ip: IP address for RAN metrics connection.
        ran_metrics_port: Port for RAN metrics connection.
        ran_control_ip: IP address for RAN control connection.
        ran_control_port: Port for RAN control connection.
        enable_logging: Whether to enable file logging.
        window_size: Size of the sliding window for analysis.
        model_path: Path to the trained ML model file.
        scaler_path: Path to the scaler file for model preprocessing.
        min_ddl: Minimum deadline value in milliseconds.
        priority_update_interval: Interval for priority updates in seconds.
        slot_duration_ms: Duration of each slot in milliseconds.
    """
    slo_ctrl_port: int = 5557
    ran_metrics_ip: str = "127.0.0.1"
    ran_metrics_port: int = 5556
    ran_control_ip: str = "127.0.0.1"
    ran_control_port: int = 5555
    enable_logging: bool = False
    window_size: int = 5
    model_path: Optional[str] = None
    scaler_path: Optional[str] = None
    min_ddl: float = 0.1
    priority_update_interval: float = 0.001
    slot_duration_ms: float = 0.5


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
    RNTI_STRING_LENGTH = 4
    SLOT_MAX = 20480
    RECV_BUFFER_SIZE = 1024
    SOCKET_LISTEN_BACKLOG = 5


# Message types
class MessageTypes:
    """Constants for message types."""
    NEW_UE = "NEW_UE"
    
    # RAN metrics types
    PRB = "PRB"
    SR = "SR"  
    BSR = "BSR"


# Default file paths
class DefaultPaths:
    """Default file paths for model and data files."""
    MODEL_PATH = "decision_tree/models/bsr_only_xgboost.joblib"
    SCALER_PATH = "decision_tree/models/bsr_only_scaler.joblib"
    LOG_FILE = "controller.log" 