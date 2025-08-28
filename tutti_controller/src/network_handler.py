"""Network communication handler for Tutti Controller."""

import socket
import struct
from typing import Optional, Tuple, Any

from .config import ControllerConfig, DefaultValues
from .utils import Logger


class NetworkHandler:
    """Handles all network communication for the Tutti Controller.
    
    This class manages UDP sockets for receiving application messages,
    receiving RAN metrics, and sending priority control messages.
    """
    
    def __init__(self, config: ControllerConfig, logger: Logger):
        """Initialize the network handler.
        
        Args:
            config: Controller configuration settings.
            logger: Logger instance for writing events.
        """
        self.config = config
        self.logger = logger
        
        # Socket instances
        self.app_socket: Optional[socket.socket] = None
        self.ran_metrics_socket: Optional[socket.socket] = None
        self.ran_control_socket: Optional[socket.socket] = None
    
    def initialize_sockets(self) -> bool:
        """Initialize all UDP sockets.
        
        Returns:
            True if all sockets were initialized successfully, False otherwise.
        """
        try:
            self._setup_app_socket()
            self._setup_ran_metrics_socket()
            self._setup_ran_control_socket()
            return True
        except Exception as e:
            self.logger.write(f"Failed to initialize sockets: {e}\n")
            self.logger.flush()
            return False
    
    def _setup_app_socket(self) -> None:
        """Set up UDP socket for receiving application messages."""
        self.app_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.app_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.app_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.app_socket.bind(("0.0.0.0", self.config.app_port))
        
        self.logger.write(f"Application UDP server listening on port {self.config.app_port}\n")
        self.logger.flush()
    
    def _setup_ran_metrics_socket(self) -> None:
        """Set up UDP socket for receiving RAN metrics."""
        self.ran_metrics_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ran_metrics_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.ran_metrics_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.ran_metrics_socket.bind((self.config.ran_metrics_ip, self.config.ran_metrics_port))
        
        self.logger.write(
            f"RAN metrics UDP server listening on {self.config.ran_metrics_ip}:"
            f"{self.config.ran_metrics_port}\n"
        )
        self.logger.flush()
    
    def _setup_ran_control_socket(self) -> None:
        """Set up UDP socket for sending RAN control messages."""
        self.ran_control_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ran_control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.ran_control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        
        self.logger.write(
            f"RAN control UDP sender initialized (target: {self.config.ran_control_ip}:"
            f"{self.config.ran_control_port})\n"
        )
        self.logger.flush()
    
    def receive_app_message(self) -> Optional[Tuple[bytes, Any]]:
        """Receive an application message from the app socket.
        
        Returns:
            Tuple of (data, address) if message received, None otherwise.
        """
        try:
            if self.app_socket:
                return self.app_socket.recvfrom(DefaultValues.SOCKET_BUFFER_SIZE)
        except Exception as e:
            self.logger.write(f"Error receiving application message: {e}\n")
            self.logger.flush()
        return None
    
    def receive_ran_metrics(self) -> Optional[Tuple[bytes, Any]]:
        """Receive RAN metrics from the metrics socket.
        
        Returns:
            Tuple of (data, address) if message received, None otherwise.
        """
        try:
            if self.ran_metrics_socket:
                return self.ran_metrics_socket.recvfrom(DefaultValues.SOCKET_BUFFER_SIZE)
        except Exception as e:
            self.logger.write(f"Error receiving RAN metrics: {e}\n")
            self.logger.flush()
        return None
    
    def send_priority_update(self, rnti: int, priority: float, is_reset: bool = False) -> bool:
        """Send priority update to RAN via UDP.
        
        Args:
            rnti: The RNTI as integer value.
            priority: Priority value to set.
            is_reset: Whether this is a priority reset message.
            
        Returns:
            True if message was sent successfully, False otherwise.
        """
        try:
            if not self.ran_control_socket:
                return False
            
            # Pack message: RNTI as int, priority as double, reset as bool
            message = struct.pack("=IdB", rnti, priority, is_reset)
            
            # Send via UDP to target address
            self.ran_control_socket.sendto(
                message,
                (self.config.ran_control_ip, self.config.ran_control_port)
            )
            return True
        except Exception as e:
            self.logger.write(f"Failed to send priority update: {e}\n")
            self.logger.flush()
            return False
    
    def close_sockets(self) -> None:
        """Close all network sockets."""
        sockets_to_close = [
            self.app_socket,
            self.ran_metrics_socket,
            self.ran_control_socket
        ]
        
        for sock in sockets_to_close:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass  # Ignore errors during cleanup
        
        self.app_socket = None
        self.ran_metrics_socket = None
        self.ran_control_socket = None