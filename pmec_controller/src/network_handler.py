"""Network communication handling for PMEC Controller."""

import socket
import struct
import threading
from typing import Dict, Optional, Callable

from .config import ControllerConfig, MessageTypes, NetworkConstants
from .utils import Logger, format_rnti_string
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
        self.metrics_callback: Optional[Callable[[str], None]] = None
    
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
            self.slo_ctrl_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.slo_ctrl_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.slo_ctrl_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            self.slo_ctrl_socket.bind(("0.0.0.0", self.config.slo_ctrl_port))
            self.slo_ctrl_socket.listen(NetworkConstants.SOCKET_LISTEN_BACKLOG)
            
            self.logger.log(f"SLO control plane server listening on port {self.config.slo_ctrl_port}")
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
    
    def set_metrics_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback function for RAN metrics processing.
        
        Args:
            callback: Function to call when RAN metrics are received.
        """
        self.metrics_callback = callback
    
    def _handle_slo_ctrl_connections(self) -> None:
        """Handle incoming SLO control plane connections."""
        while self.running and self.slo_ctrl_socket:
            try:
                conn, addr = self.slo_ctrl_socket.accept()
                self.logger.log(f"New SLO control plane connection from {addr}")
                
                threading.Thread(
                    target=self._handle_slo_ctrl_messages,
                    args=(conn, addr),
                    daemon=True,
                ).start()
                
            except Exception as e:
                if self.running:  # Only log if we're still supposed to be running
                    self.logger.log(f"Error accepting SLO control plane connection: {e}")
    
    def _handle_slo_ctrl_messages(self, conn: socket.socket, addr) -> None:
        """Handle messages from a specific SLO control plane connection.
        
        Args:
            conn: Socket connection to the SLO control plane.
            addr: Address of the SLO control plane client.
        """
        try:
            self.logger.log(f"SLO control plane client connected from {addr}")

            while self.running:
                try:
                    data = conn.recv(NetworkConstants.RECV_BUFFER_SIZE)
                    if not data:
                        break

                    message = data.decode("utf-8").strip()
                    self._process_slo_ctrl_message(message)

                except Exception as e:
                    self.logger.log(f"Error processing SLO control plane message: {e}")
                    break

        except Exception as e:
            self.logger.log(f"Error in SLO control plane message handler: {e}")
        finally:
            conn.close()
    
    def _process_slo_ctrl_message(self, message: str) -> None:
        """Process a message from SLO control plane.
        
        Args:
            message: The message string to process.
        """
        try:
            msg_parts = message.split("|")
            msg_type = msg_parts[0]

            if msg_type == MessageTypes.NEW_UE:
                self._handle_new_ue_message(msg_parts)
            else:
                self.logger.log(f"Unknown message type: {msg_type}")
                
        except Exception as e:
            self.logger.log(f"Error processing SLO control plane message '{message}': {e}")
    
    def _handle_new_ue_message(self, msg_parts: list) -> None:
        """Handle NEW_UE message from SLO control plane.
        
        Args:
            msg_parts: Message parts split by '|'.
        """
        if len(msg_parts) != 3:
            self.logger.log(f"Invalid NEW_UE message format: {msg_parts}")
            return
        
        _, rnti, slo_latency = msg_parts
        
        self.priority_manager.register_ue(
            rnti, 
            float(slo_latency)
        )

    
    def _handle_ran_metrics(self) -> None:
        """Receive and process RAN metrics."""
        while self.running and self.ran_metrics_socket:
            try:
                data = self.ran_metrics_socket.recv(
                    NetworkConstants.RECV_BUFFER_SIZE
                ).decode("utf-8")
                
                if not data:
                    continue

                # Forward to metrics processor via callback
                if self.metrics_callback:
                    self.metrics_callback(data)

            except Exception as e:
                if self.running:  # Only log if we're still supposed to be running
                    self.logger.log(f"Error receiving RAN metrics: {e}")
    
    def send_priority_update(self, rnti: str, priority: float, reset: bool = False) -> bool:
        """Send priority update to RAN.
        
        Args:
            rnti: Radio Network Temporary Identifier.
            priority: Priority value to set.
            reset: Whether this is a priority reset.
            
        Returns:
            True if update was sent successfully.
        """
        try:
            if not self.ran_control_socket:
                self.logger.log("RAN control socket not available")
                return False
            
            # Format RNTI string and pack message
            rnti_bytes = format_rnti_string(rnti, NetworkConstants.RNTI_STRING_LENGTH)
            msg = struct.pack("=5sdb", rnti_bytes, priority, reset)

            self.ran_control_socket.send(msg)
            return True
            
        except Exception as e:
            self.logger.log(f"Failed to send priority update: {e}")
            return False
    
    def set_priority(self, rnti: str, priority: float) -> bool:
        """Send priority update to RAN.
        
        Args:
            rnti: Radio Network Temporary Identifier.
            priority: Priority value to set.
            
        Returns:
            True if update was sent successfully.
        """
        return self.send_priority_update(rnti, priority, reset=False)
    
    def reset_priority(self, rnti: str) -> bool:
        """Reset priority for a specific RNTI.
        
        Args:
            rnti: Radio Network Temporary Identifier.
            
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
    
    def get_connection_status(self) -> Dict[str, any]:
        """Get status of all network connections.
        
        Returns:
            Dictionary with connection status information.
        """
        return {
            "running": self.running,
            "slo_ctrl_server_active": self.slo_ctrl_socket is not None,
            "ran_metrics_connected": self.ran_metrics_socket is not None,
            "ran_control_connected": self.ran_control_socket is not None,
            "active_slo_ctrl_connections": 0 # No longer tracking active SLO control plane connections
        } 