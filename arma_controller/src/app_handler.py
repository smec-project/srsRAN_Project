"""Application message handler for Tutti Controller."""

import struct
import time
from typing import Dict, Optional

from .config import ControllerConfig, DefaultValues
from .utils import Logger, rnti_to_hex_string
from .network_handler import NetworkHandler


class AppHandler:
    """Handles application messages and UE registration.
    
    This class processes UDP messages from applications, manages UE information,
    and tracks request states and timing.
    """
    
    def __init__(self, config: ControllerConfig, logger: Logger, network_handler: NetworkHandler):
        """Initialize the application handler.
        
        Args:
            config: Controller configuration settings.
            logger: Logger instance for writing events.
            network_handler: Network handler for receiving messages.
        """
        self.config = config
        self.logger = logger
        self.network_handler = network_handler
        
        # UE tracking data structures
        self.ue_info: Dict[str, dict] = {}  # RNTI (str) -> UE info
        self.ue_resource_needs: Dict[str, int] = {}  # RNTI -> total bytes needed
        self.ue_pending_requests: Dict[str, Dict[int, int]] = {}  # RNTI -> {request_id -> bytes}
        self.request_start_times: Dict[str, Dict[int, float]] = {}  # RNTI -> {request_id -> start_time}
        self.last_completed_request_id: Dict[str, int] = {}  # RNTI -> last completed request ID
    
    def process_app_messages(self) -> None:
        """Process incoming application messages continuously."""
        while True:  # This will be controlled by the main controller's running flag
            message_data = self.network_handler.receive_app_message()
            if message_data:
                data, addr = message_data
                self._process_udp_message(data, addr)
    
    def _process_udp_message(self, data: bytes, addr) -> None:
        """Process a single UDP message.
        
        Args:
            data: Raw UDP message data.
            addr: Source address of the message.
        """
        if not data:
            return
        
        # Validate message size (expect 20 bytes: 5 x 32-bit integers)
        if len(data) != DefaultValues.MESSAGE_SIZE:
            self.logger.write(f"Invalid message size: {len(data)} bytes, expected {DefaultValues.MESSAGE_SIZE}\n")
            self.logger.flush()
            return
        
        # Unpack message: rnti, request_index, request_size, slo_ms, start_or_end
        rnti, request_index, request_size, slo_ms, start_or_end = struct.unpack('IIIII', data)
        
        # Convert RNTI to string for compatibility
        rnti_str = rnti_to_hex_string(rnti)
        
        # Process the message based on start_or_end flag
        self._handle_request_message(rnti_str, request_index, request_size, slo_ms, start_or_end)
    
    def _handle_request_message(
        self, 
        rnti_str: str, 
        request_index: int, 
        request_size: int, 
        slo_ms: int, 
        start_or_end: int
    ) -> None:
        """Handle a request start or end message.
        
        Args:
            rnti_str: RNTI as hex string.
            request_index: Index/ID of the request.
            request_size: Size of the request in bytes.
            slo_ms: Service Level Objective in milliseconds.
            start_or_end: 0 for start, 1 for end.
        """
        try:
            if start_or_end == 0:  # Request START
                self._handle_request_start(rnti_str, request_index, request_size, slo_ms)
            elif start_or_end == 1:  # Request END
                self._handle_request_end(rnti_str, request_index)
            else:
                self.logger.write(f"Invalid start_or_end value: {start_or_end}\n")
                self.logger.flush()
        except Exception as e:
            self.logger.write(f"Error processing UDP app message: {e}\n")
            self.logger.flush()
    
    def _handle_request_start(
        self, 
        rnti_str: str, 
        request_index: int, 
        request_size: int, 
        slo_ms: int
    ) -> None:
        """Handle the start of a new request.
        
        Args:
            rnti_str: RNTI as hex string.
            request_index: Index/ID of the request.
            request_size: Size of the request in bytes.
            slo_ms: Service Level Objective in milliseconds.
        """
        # Auto-register UE if not exists
        if rnti_str not in self.ue_info:
            self._register_ue(rnti_str, request_size, slo_ms)
        
        # Start new request
        self.ue_resource_needs[rnti_str] += request_size
        self.ue_pending_requests[rnti_str][request_index] = request_size
        
        # Record start time
        if rnti_str not in self.request_start_times:
            self.request_start_times[rnti_str] = {}
        self.request_start_times[rnti_str][request_index] = time.time()
        
        self.logger.write(
            f"Request {request_index} from RNTI {rnti_str} started at {time.time()}\n"
        )
        self.logger.flush()
    
    def _handle_request_end(self, rnti_str: str, request_index: int) -> None:
        """Handle the end of a request.
        
        Args:
            rnti_str: RNTI as hex string.
            request_index: Index/ID of the request.
        """
        # Calculate and log completion time
        if (rnti_str in self.request_start_times and 
            request_index in self.request_start_times[rnti_str]):
            
            start_time = self.request_start_times[rnti_str][request_index]
            elapsed_time_ms = (time.time() - start_time) * 1000
            
            self.logger.write(
                f"Request {request_index} from RNTI {rnti_str} completed in "
                f"{elapsed_time_ms:.2f}ms at {time.time()}\n"
            )
            self.logger.flush()
            
            # Update last completed request ID
            if rnti_str not in self.last_completed_request_id:
                self.last_completed_request_id[rnti_str] = request_index
            else:
                self.last_completed_request_id[rnti_str] = max(
                    self.last_completed_request_id[rnti_str], request_index
                )
            
            # Clean up completed requests
            self._cleanup_completed_requests(rnti_str, request_index)
        
        # Handle resource cleanup
        if (rnti_str in self.ue_pending_requests and 
            request_index in self.ue_pending_requests[rnti_str]):
            
            completed_size = self.ue_pending_requests[rnti_str][request_index]
            self.ue_resource_needs[rnti_str] -= completed_size
            del self.ue_pending_requests[rnti_str][request_index]
    
    def _register_ue(self, rnti_str: str, request_size: int, slo_ms: int) -> None:
        """Register a new UE with auto-detected parameters.
        
        Args:
            rnti_str: RNTI as hex string.
            request_size: Size of the first request in bytes.
            slo_ms: Service Level Objective in milliseconds.
        """
        self.ue_info[rnti_str] = {
            "app_id": "udp_app",
            "ue_idx": "auto",
            "latency_req": float(slo_ms),
            "request_size": request_size,
        }
        self.ue_resource_needs[rnti_str] = 0
        self.ue_pending_requests[rnti_str] = {}
        self.request_start_times[rnti_str] = {}
        
        self.logger.write(
            f"Auto-registered UE - RNTI: {rnti_str}, SLO: {slo_ms}ms, "
            f"Size: {request_size} bytes at {time.time()}\n"
        )
        self.logger.flush()
    
    def _cleanup_completed_requests(self, rnti_str: str, request_index: int) -> None:
        """Clean up all request data for completed requests.
        
        Args:
            rnti_str: RNTI as hex string.
            request_index: Index/ID of the completed request.
        """
        # Clean up all requests <= current request_index
        if rnti_str in self.request_start_times:
            expired_req_ids = [
                req_id for req_id in self.request_start_times[rnti_str] 
                if req_id <= request_index
            ]
            for req_id in expired_req_ids:
                if req_id in self.request_start_times[rnti_str]:
                    del self.request_start_times[rnti_str][req_id]
    
    def get_ue_info(self, rnti_str: str) -> Optional[dict]:
        """Get UE information for a given RNTI.
        
        Args:
            rnti_str: RNTI as hex string.
            
        Returns:
            UE information dictionary or None if not found.
        """
        return self.ue_info.get(rnti_str)
    
    def get_pending_requests(self, rnti_str: str) -> Dict[int, int]:
        """Get pending requests for a UE.
        
        Args:
            rnti_str: RNTI as hex string.
            
        Returns:
            Dictionary of request_id -> request_size.
        """
        return self.ue_pending_requests.get(rnti_str, {})
    
    def get_request_start_times(self, rnti_str: str) -> Dict[int, float]:
        """Get request start times for a UE.
        
        Args:
            rnti_str: RNTI as hex string.
            
        Returns:
            Dictionary of request_id -> start_time.
        """
        return self.request_start_times.get(rnti_str, {})
    
    def get_last_completed_request_id(self, rnti_str: str) -> int:
        """Get the last completed request ID for a UE.
        
        Args:
            rnti_str: RNTI as hex string.
            
        Returns:
            Last completed request ID, or -1 if none.
        """
        return self.last_completed_request_id.get(rnti_str, -1)