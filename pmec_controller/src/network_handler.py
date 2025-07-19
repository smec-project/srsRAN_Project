"""Network communication handling for PMEC Controller."""

import socket
import struct
import threading
from typing import Dict, Optional, Callable, Any

from .config import ControllerConfig, MessageTypes, NetworkConstants
from .utils import Logger
from .priority_manager import PriorityManager


class NetworkHandler:
    """Handles network communications for SLO control plane and RAN connections.
    
    Manages SLO control plane server connections, RAN metrics reception,
    and RAN control commands for priority updates.
    """
    
    def __init__(
        self,
        config: ControllerConfig,
        priority_manager: PriorityManager,
        logger: Logger
    ):
        """Initialize the network handler.
        
        Args:
            config: Configuration settings.
            priority_manager: Priority manager instance.
            logger: Logger instance for debugging output.
        """
        self.config = config
        self.priority_manager = priority_manager
        self.logger = logger
        
        # Running state
        self.running = False
        
        # SLO control plane server setup
        self.slo_ctrl_socket: Optional[socket.socket] = None
        
        # RAN connections
        self.ran_metrics_socket: Optional[socket.socket] = None
        self.ran_control_socket: Optional[socket.socket] = None
        
        # Callback for RAN metrics processing
        self.metrics_callback: Optional[Callable[[bytes], None]] = None
    
    def setup_connections(self) -> bool:
        """Set up all network connections.
        
        Returns:
            True if all connections were set up successfully.
        """
        try:
            # Setup SLO control plane server
            if not self._setup_slo_ctrl_server():
                return False
            
            # Setup RAN connections
            if not self._setup_ran_connections():
                return False
            
            self.logger.log("All network connections set up successfully")
            return True
            
        except Exception as e:
            self.logger.log(f"Error setting up network connections: {e}")
            return False
    
    def _setup_slo_ctrl_server(self) -> bool:
        """Set up the SLO control plane server socket.
        
        Returns:
            True if setup was successful.
        """
        try:
            self.slo_ctrl_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.slo_ctrl_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.slo_ctrl_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            self.slo_ctrl_socket.bind(("0.0.0.0", self.config.slo_ctrl_port))
            
            self.logger.log(f"SLO control plane server listening on port {self.config.slo_ctrl_port} (UDP)")
            return True
            
        except Exception as e:
            self.logger.log(f"Error setting up SLO control plane server: {e}")
            return False
    
    def _setup_ran_connections(self) -> bool:
        """Set up RAN metrics and control connections.
        
        Returns:
            True if setup was successful.
        """
        try:
            # RAN metrics connection
            self.ran_metrics_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.ran_metrics_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.ran_metrics_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            
            # RAN control connection
            self.ran_control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.ran_control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.ran_control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            
            # Connect to RAN services
            self.ran_metrics_socket.connect(
                (self.config.ran_metrics_ip, self.config.ran_metrics_port)
            )
            self.ran_control_socket.connect(
                (self.config.ran_control_ip, self.config.ran_control_port)
            )
            
            self.logger.log(f"Connected to RAN metrics at {self.config.ran_metrics_ip}:{self.config.ran_metrics_port}")
            self.logger.log(f"Connected to RAN control at {self.config.ran_control_ip}:{self.config.ran_control_port}")
            return True
            
        except Exception as e:
            self.logger.log(f"Error setting up RAN connections: {e}")
            return False
    
    def start_networking(self) -> bool:
        """Start all networking threads.
        
        Returns:
            True if networking was started successfully.
        """
        if not self.setup_connections():
            return False
        
        self.running = True
        
        # Start networking threads
        threading.Thread(target=self._handle_slo_ctrl_connections, daemon=True).start()
        threading.Thread(target=self._handle_ran_metrics, daemon=True).start()
        
        self.logger.log("Network handling threads started")
        return True
    
    def set_metrics_callback(self, callback: Callable[[bytes], None]) -> None:
        """Set callback function for RAN metrics processing.
        
        Args:
            callback: Function to call when RAN metrics are received.
        """
        self.metrics_callback = callback
    
    def _handle_slo_ctrl_connections(self) -> None:
        """Handle incoming SLO control plane messages (UDP)."""
        while self.running and self.slo_ctrl_socket:
            try:
                data, addr = self.slo_ctrl_socket.recvfrom(NetworkConstants.RECV_BUFFER_SIZE)
                self.logger.log(f"SLO control plane message from {addr}")
                
                # Process message directly (no threading needed for UDP)
                self._process_slo_ctrl_message(data)
                
            except Exception as e:
                if self.running:  # Only log if we're still supposed to be running
                    self.logger.log(f"Error receiving SLO control plane message: {e}")
    
    def _handle_slo_ctrl_messages(self, data: bytes, addr) -> None:
        """Process SLO control plane message (UDP version).
        
        Args:
            data: The binary message data to process.
            addr: Address of the SLO control plane client.
        """
        try:
            self.logger.log(f"Processing SLO control plane message from {addr}")
            self._process_slo_ctrl_message(data)
            
        except Exception as e:
            self.logger.log(f"Error processing SLO control plane message: {e}")
    
    def _process_slo_ctrl_message(self, data: bytes) -> None:
        """Process a binary message from SLO control plane.
        
        Args:
            data: The binary message data to process.
        """
        try:
            # Expect 12 bytes: uint32 (message_type) + uint32 (rnti) + uint32 (slo_latency)
            if len(data) != 12:
                self.logger.log(f"Invalid SLO control message size: {len(data)} bytes, expected 12")
                return
            
            # Unpack: uint32 message_type, uint32 RNTI, uint32 SLO latency
            msg_type, rnti, slo_latency_uint = struct.unpack('=III', data)
            
            if msg_type == MessageTypes.SLO_MESSAGE:
                # Convert uint32 SLO latency to float
                slo_latency_float = float(slo_latency_uint)
                self.priority_manager.register_ue(rnti, slo_latency_float)
            else:
                self.logger.log(f"Unknown message type: {msg_type}")
            
        except Exception as e:
            self.logger.log(f"Error processing SLO control plane binary message: {e}")

    
    def _handle_ran_metrics(self) -> None:
        """Receive and process binary RAN metrics."""
        while self.running and self.ran_metrics_socket:
            try:
                data = self.ran_metrics_socket.recv(
                    NetworkConstants.RECV_BUFFER_SIZE
                )
                
                if not data:
                    continue

                # Forward binary data to metrics processor via callback
                if self.metrics_callback:
                    self.metrics_callback(data)

            except Exception as e:
                if self.running:  # Only log if we're still supposed to be running
                    self.logger.log(f"Error receiving RAN metrics: {e}")
    
    def send_priority_update(self, rnti: int, priority: float, reset: bool = False) -> bool:
        """Send priority update to RAN.
        
        Args:
            rnti: Radio Network Temporary Identifier as integer.
            priority: Priority value to set.
            reset: Whether this is a priority reset.
            
        Returns:
            True if update was sent successfully.
        """
        try:
            if not self.ran_control_socket:
                self.logger.log("RAN control socket not available")
                return False
            
            # Pack message with RNTI as integer
            msg = struct.pack("=IdB", rnti, priority, reset)

            self.ran_control_socket.send(msg)
            return True
            
        except Exception as e:
            self.logger.log(f"Failed to send priority update: {e}")
            return False
    
    def set_priority(self, rnti: int, priority: float) -> bool:
        """Send priority update to RAN.
        
        Args:
            rnti: Radio Network Temporary Identifier as integer.
            priority: Priority value to set.
            
        Returns:
            True if update was sent successfully.
        """
        return self.send_priority_update(rnti, priority, reset=False)
    
    def reset_priority(self, rnti: int) -> bool:
        """Reset priority for a specific RNTI.
        
        Args:
            rnti: Radio Network Temporary Identifier as integer.
            
        Returns:
            True if reset was sent successfully.
        """
        return self.send_priority_update(rnti, 0.0, reset=True)
    
    def stop_networking(self) -> None:
        """Stop all networking operations and clean up connections."""
        self.running = False

        # Close all SLO control plane connections
        # The slo_ctrl_connections dictionary is removed, so this loop is no longer needed.

        # Close server sockets
        sockets_to_close = [
            self.slo_ctrl_socket,
            self.ran_metrics_socket,
            self.ran_control_socket
        ]

        for sock in sockets_to_close:
            if sock:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                    sock.close()
                except:
                    pass

        self.logger.log("Network connections closed")
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get status of all network connections.
        
        Returns:
            Dictionary with connection status information.
        """
        return {
            "running": self.running,
            "slo_ctrl_server_active": self.slo_ctrl_socket is not None,
            "ran_metrics_connected": self.ran_metrics_socket is not None,
            "ran_control_connected": self.ran_control_socket is not None,
            "active_slo_ctrl_connections": "N/A (UDP)"  # UDP doesn't maintain connections
        } 