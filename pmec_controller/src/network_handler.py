"""Network communication handling for PMEC Controller."""

import socket
import struct
import threading
import time
from typing import Dict, Optional, Callable

from .config import ControllerConfig, MessageTypes, NetworkConstants
from .utils import Logger, format_rnti_string, get_current_timestamp, calculate_elapsed_time_ms
from .priority_manager import PriorityManager


class NetworkHandler:
    """Handles network communications for application and RAN connections.
    
    Manages application server connections, RAN metrics reception,
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
        
        # Application server setup
        self.app_socket: Optional[socket.socket] = None
        self.app_connections: Dict[str, socket.socket] = {}
        
        # RAN connections
        self.ran_metrics_socket: Optional[socket.socket] = None
        self.ran_control_socket: Optional[socket.socket] = None
        
        # Request tracking for applications
        self.request_start_times: Dict[str, Dict[int, float]] = {}
        
        # Callback for RAN metrics processing
        self.metrics_callback: Optional[Callable[[str], None]] = None
    
    def setup_connections(self) -> bool:
        """Set up all network connections.
        
        Returns:
            True if all connections were set up successfully.
        """
        try:
            # Setup application server
            if not self._setup_app_server():
                return False
            
            # Setup RAN connections
            if not self._setup_ran_connections():
                return False
            
            self.logger.log("All network connections set up successfully")
            return True
            
        except Exception as e:
            self.logger.log(f"Error setting up network connections: {e}")
            return False
    
    def _setup_app_server(self) -> bool:
        """Set up the application server socket.
        
        Returns:
            True if setup was successful.
        """
        try:
            self.app_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.app_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.app_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            self.app_socket.bind(("0.0.0.0", self.config.app_port))
            self.app_socket.listen(NetworkConstants.SOCKET_LISTEN_BACKLOG)
            
            self.logger.log(f"Application server listening on port {self.config.app_port}")
            return True
            
        except Exception as e:
            self.logger.log(f"Error setting up application server: {e}")
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
        threading.Thread(target=self._handle_app_connections, daemon=True).start()
        threading.Thread(target=self._handle_ran_metrics, daemon=True).start()
        
        self.logger.log("Network handling threads started")
        return True
    
    def set_metrics_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback function for RAN metrics processing.
        
        Args:
            callback: Function to call when RAN metrics are received.
        """
        self.metrics_callback = callback
    
    def _handle_app_connections(self) -> None:
        """Handle incoming application connections."""
        while self.running and self.app_socket:
            try:
                conn, addr = self.app_socket.accept()
                self.logger.log(f"New application connected from {addr}")
                
                threading.Thread(
                    target=self._handle_app_messages,
                    args=(conn, addr),
                    daemon=True,
                ).start()
                
            except Exception as e:
                if self.running:  # Only log if we're still supposed to be running
                    self.logger.log(f"Error accepting application connection: {e}")
    
    def _handle_app_messages(self, conn: socket.socket, addr) -> None:
        """Handle messages from a specific application connection.
        
        Args:
            conn: Socket connection to the application.
            addr: Address of the application.
        """
        app_id = None
        try:
            # First message should be app registration with app_id
            data = conn.recv(NetworkConstants.RECV_BUFFER_SIZE)
            if not data:
                return

            app_id = data.decode("utf-8").strip()
            self.app_connections[app_id] = conn
            self.logger.log(f"Application {app_id} registered from {addr}")

            while self.running:
                try:
                    data = conn.recv(NetworkConstants.RECV_BUFFER_SIZE)
                    if not data:
                        break

                    message = data.decode("utf-8").strip()
                    self._process_app_message(message, app_id)

                except Exception as e:
                    self.logger.log(f"Error processing application message: {e}")
                    break

        except Exception as e:
            self.logger.log(f"Error in application message handler: {e}")
        finally:
            if app_id and app_id in self.app_connections:
                del self.app_connections[app_id]
            conn.close()
    
    def _process_app_message(self, message: str, app_id: str) -> None:
        """Process a message from an application.
        
        Args:
            message: The message string to process.
            app_id: Application identifier.
        """
        try:
            msg_parts = message.split("|")
            msg_type = msg_parts[0]

            if msg_type == MessageTypes.NEW_UE:
                self._handle_new_ue_message(msg_parts, app_id)
            elif msg_type == MessageTypes.START:
                self._handle_start_message(msg_parts)
            elif msg_type == MessageTypes.REQUEST:
                self._handle_request_message(msg_parts)
            elif msg_type == MessageTypes.DONE:
                self._handle_done_message(msg_parts)
            else:
                self.logger.log(f"Unknown message type: {msg_type}")
                
        except Exception as e:
            self.logger.log(f"Error processing app message '{message}': {e}")
    
    def _handle_new_ue_message(self, msg_parts: list, app_id: str) -> None:
        """Handle NEW_UE message from application.
        
        Args:
            msg_parts: Message parts split by '|'.
            app_id: Application identifier.
        """
        if len(msg_parts) != 5:
            self.logger.log(f"Invalid NEW_UE message format: {msg_parts}")
            return
        
        _, rnti, ue_idx, latency_req, request_size = msg_parts
        
        self.priority_manager.register_ue(
            rnti, app_id, ue_idx, 
            float(latency_req), int(request_size)
        )
        
        # Initialize request tracking
        self.request_start_times[rnti] = {}
    
    def _handle_start_message(self, msg_parts: list) -> None:
        """Handle Start message from application.
        
        Args:
            msg_parts: Message parts split by '|'.
        """
        if len(msg_parts) != 3:
            self.logger.log(f"Invalid Start message format: {msg_parts}")
            return
        
        _, rnti, seq_num = msg_parts
        current_time = get_current_timestamp()
        
        self.logger.log(
            f"Request {seq_num} from RNTI {rnti} start at {current_time}"
        )
    
    def _handle_request_message(self, msg_parts: list) -> None:
        """Handle REQUEST message from application.
        
        Args:
            msg_parts: Message parts split by '|'.
        """
        if len(msg_parts) != 3:
            self.logger.log(f"Invalid REQUEST message format: {msg_parts}")
            return
        
        _, rnti, seq_num = msg_parts
        seq_num = int(seq_num)

        if rnti in self.priority_manager.ue_info:
            # Store request start time
            if rnti not in self.request_start_times:
                self.request_start_times[rnti] = {}
            self.request_start_times[rnti][seq_num] = get_current_timestamp()
        else:
            self.logger.log(f"Warning: Request for unknown RNTI {rnti}")
    
    def _handle_done_message(self, msg_parts: list) -> None:
        """Handle DONE message from application.
        
        Args:
            msg_parts: Message parts split by '|'.
        """
        if len(msg_parts) != 3:
            self.logger.log(f"Invalid DONE message format: {msg_parts}")
            return
        
        _, rnti, seq_num = msg_parts
        seq_num = int(seq_num)

        # Calculate final processing time
        if (rnti in self.request_start_times and 
            seq_num in self.request_start_times[rnti]):
            
            start_time = self.request_start_times[rnti][seq_num]
            elapsed_time_ms = calculate_elapsed_time_ms(start_time)
            
            self.logger.log(
                f"Request {seq_num} from RNTI {rnti} completed "
                f"in {elapsed_time_ms:.2f}ms at {get_current_timestamp()}"
            )
            
            del self.request_start_times[rnti][seq_num]
    
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

        # Close all application connections
        for conn in self.app_connections.values():
            try:
                conn.shutdown(socket.SHUT_RDWR)
                conn.close()
            except:
                pass
        self.app_connections.clear()

        # Close server sockets
        sockets_to_close = [
            self.app_socket,
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
    
    def get_connection_status(self) -> Dict[str, bool]:
        """Get status of all network connections.
        
        Returns:
            Dictionary with connection status information.
        """
        return {
            "running": self.running,
            "app_server_active": self.app_socket is not None,
            "ran_metrics_connected": self.ran_metrics_socket is not None,
            "ran_control_connected": self.ran_control_socket is not None,
            "active_app_connections": len(self.app_connections)
        } 