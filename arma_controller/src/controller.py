"""Main Tutti Controller class that orchestrates all components."""

import threading
import time
from typing import Optional

from .config import ControllerConfig
from .utils import Logger
from .network_handler import NetworkHandler
from .app_handler import AppHandler
from .metrics_processor import MetricsProcessor
from .priority_manager import PriorityManager


class TuttiController:
    """Main Tutti Controller class.
    
    Orchestrates all components including network handling, application message
    processing, RAN metrics processing, and priority management for UDP-based
    5G RAN traffic scheduling.
    """
    
    def __init__(self, config: Optional[ControllerConfig] = None):
        """Initialize the Tutti Controller.
        
        Args:
            config: Configuration settings. Uses default if None.
        """
        self.config = config or ControllerConfig()
        self.running = False
        
        # Initialize logger
        self.logger = Logger(self.config.log_file_path)
        
        # Initialize components in dependency order
        self.network_handler = NetworkHandler(self.config, self.logger)
        self.app_handler = AppHandler(self.config, self.logger, self.network_handler)
        self.metrics_processor = MetricsProcessor(self.config, self.logger, self.network_handler)
        self.priority_manager = PriorityManager(
            self.config,
            self.logger,
            self.network_handler,
            self.app_handler,
            self.metrics_processor
        )
        
        # Thread references
        self.app_thread: Optional[threading.Thread] = None
        self.metrics_thread: Optional[threading.Thread] = None
    
    def start(self) -> bool:
        """Start the controller and all its components.
        
        Returns:
            True if started successfully, False otherwise.
        """
        self.running = True
        
        # Initialize network sockets
        if not self.network_handler.initialize_sockets():
            self.logger.write("Failed to initialize network sockets\n")
            self.logger.flush()
            return False
        
        # Start processing threads
        self.app_thread = threading.Thread(
            target=self._handle_app_messages, 
            daemon=True
        )
        self.metrics_thread = threading.Thread(
            target=self._handle_ran_metrics, 
            daemon=True
        )
        
        self.app_thread.start()
        self.metrics_thread.start()
        
        self.logger.write("Tutti Controller started successfully\n")
        self.logger.flush()
        return True
    
    def stop(self) -> None:
        """Stop the controller and clean up resources."""
        self.running = False
        
        # Close network connections
        self.network_handler.close_sockets()
        
        # Close logger
        self.logger.close()
        
        self.logger.write("Tutti Controller stopped\n")
        self.logger.flush()
    
    def _handle_app_messages(self) -> None:
        """Handle incoming UDP application messages in a loop."""
        while self.running:
            try:
                message_data = self.network_handler.receive_app_message()
                if message_data:
                    data, addr = message_data
                    self._process_app_message(data, addr)
            except Exception as e:
                if self.running:  # Only log if we're still supposed to be running
                    self.logger.write(f"Error in app message handling: {e}\n")
                    self.logger.flush()
    
    def _handle_ran_metrics(self) -> None:
        """Handle incoming RAN metrics in a loop."""
        while self.running:
            try:
                message_data = self.network_handler.receive_ran_metrics()
                if message_data:
                    data, _ = message_data
                    self._process_ran_metrics(data)
            except Exception as e:
                if self.running:  # Only log if we're still supposed to be running
                    self.logger.write(f"Error in RAN metrics handling: {e}\n")
                    self.logger.flush()
    
    def _process_app_message(self, data: bytes, addr) -> None:
        """Process a single application message.
        
        Args:
            data: Raw UDP message data.
            addr: Source address of the message.
        """
        try:
            if not data or len(data) != 20:  # Expected message size
                return
            
            import struct
            from .utils import rnti_to_hex_string
            
            # Unpack message: rnti, request_index, request_size, slo_ms, start_or_end
            rnti, request_index, request_size, slo_ms, start_or_end = struct.unpack('IIIII', data)
            rnti_str = rnti_to_hex_string(rnti)
            
            # Process the message based on type
            if start_or_end == 0:  # Request START
                self._handle_request_start(rnti_str, request_index, request_size, slo_ms)
            elif start_or_end == 1:  # Request END
                self._handle_request_end(rnti_str, request_index)
            
        except Exception as e:
            self.logger.write(f"Error processing app message: {e}\n")
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
        if not self.app_handler.get_ue_info(rnti_str):
            self._register_ue(rnti_str, request_size, slo_ms)
        
        # Start new request
        self.app_handler.ue_resource_needs[rnti_str] += request_size
        self.app_handler.ue_pending_requests[rnti_str][request_index] = request_size
        
        if rnti_str not in self.app_handler.request_start_times:
            self.app_handler.request_start_times[rnti_str] = {}
        self.app_handler.request_start_times[rnti_str][request_index] = time.time()
        
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
        # Calculate completion time
        if (rnti_str in self.app_handler.request_start_times and 
            request_index in self.app_handler.request_start_times[rnti_str]):
            
            start_time = self.app_handler.request_start_times[rnti_str][request_index]
            elapsed_time_ms = (time.time() - start_time) * 1000
            
            self.logger.write(
                f"Request {request_index} from RNTI {rnti_str} completed in "
                f"{elapsed_time_ms:.2f}ms at {time.time()}\n"
            )
            self.logger.flush()
            
            # Update last completed request ID
            if rnti_str not in self.app_handler.last_completed_request_id:
                self.app_handler.last_completed_request_id[rnti_str] = request_index
            else:
                self.app_handler.last_completed_request_id[rnti_str] = max(
                    self.app_handler.last_completed_request_id[rnti_str], request_index
                )
            
            # Reset priority after completion
            self.priority_manager.reset_priority(rnti_str)
            
            # Clean up completed requests
            self._cleanup_completed_requests(rnti_str, request_index)
        
        # Handle resource cleanup
        if (rnti_str in self.app_handler.ue_pending_requests and 
            request_index in self.app_handler.ue_pending_requests[rnti_str]):
            
            completed_size = self.app_handler.ue_pending_requests[rnti_str][request_index]
            self.app_handler.ue_resource_needs[rnti_str] -= completed_size
            del self.app_handler.ue_pending_requests[rnti_str][request_index]
    
    def _process_ran_metrics(self, data: bytes) -> None:
        """Process RAN metrics data.
        
        Args:
            data: Raw binary metrics data.
        """
        try:
            if not data:
                return
            
            import struct
            from .utils import rnti_to_hex_string
            
            # Process binary messages (16 bytes each: 4 x 32-bit integers)
            message_size = 16
            offset = 0
            
            while offset + message_size <= len(data):
                # Unpack: msg_type, rnti, field1, field2
                msg_type, rnti, field1, field2 = struct.unpack(
                    'IIII', data[offset:offset + message_size]
                )
                
                if msg_type == 0:  # PRB allocation message
                    self._handle_prb_metrics(rnti, field1, field2)
                elif msg_type == 1:  # SR message
                    pass  # Currently not processed
                elif msg_type == 2:  # BSR message
                    self._handle_bsr_metrics(rnti, field1, field2)
                
                offset += message_size
                
        except Exception as e:
            self.logger.write(f"Error processing RAN metrics: {e}\n")
            self.logger.flush()
    
    def _handle_prb_metrics(self, rnti: int, prbs: int, slot: int) -> None:
        """Handle PRB allocation metrics.
        
        Args:
            rnti: RNTI as integer value.
            prbs: Number of PRBs allocated.
            slot: Slot number for the allocation.
        """
        rnti_str = f"{rnti:04x}"
        
        # Update metrics processor and check for slot transitions
        slot_changed = self.metrics_processor._handle_prb_metrics(rnti, prbs, slot)
        
        # Update PRB allocation for earliest active request
        if (rnti_str in self.app_handler.request_start_times and 
            self.app_handler.request_start_times[rnti_str]):
            
            earliest_req_id = min(
                self.app_handler.request_start_times[rnti_str].items(), 
                key=lambda x: x[1]
            )[0]
            
            self.metrics_processor.update_request_prb_allocation(
                rnti_str, earliest_req_id, slot, prbs
            )
        
        # Handle priority updates if slot changed
        if slot_changed:
            self._handle_slot_change_priority_updates(rnti_str, prbs)
    
    def _handle_bsr_metrics(self, rnti: int, bytes_val: int, slot: int) -> None:
        """Handle BSR metrics.
        
        Args:
            rnti: RNTI as integer value.
            bytes_val: Number of bytes reported in BSR.
            slot: Slot number for the BSR.
        """
        rnti_str = f"{rnti:04x}"
        
        # Update current metrics
        if rnti_str not in self.metrics_processor.current_metrics:
            self.metrics_processor.current_metrics[rnti_str] = {}
        
        self.metrics_processor.current_metrics[rnti_str].update({
            "BSR_BYTES": bytes_val,
            "BSR_SLOT": slot
        })
    
    
    def _handle_slot_change_priority_updates(self, current_rnti: str, prbs: int) -> None:
        """Handle priority updates when slot changes.
        
        Args:
            current_rnti: RNTI that triggered this slot update (for reference).
            prbs: PRBs allocated to current_rnti (for reference).
        """
        
        # Update RNTIs that didn't get PRBs in the previous slot
        for rnti in list(self.app_handler.request_start_times.keys()):
            if (self.app_handler.request_start_times[rnti] and 
                rnti not in self.metrics_processor.slot_prb_allocations):
                prev_slot = self.metrics_processor.current_slot - 1 if self.metrics_processor.current_slot else 0
                self.metrics_processor._update_prb_history(rnti, prev_slot, 0)
        
        # Update priorities for all RNTIs with active requests
        active_rntis = []
        for rnti in list(self.app_handler.request_start_times.keys()):
            if self.app_handler.request_start_times[rnti]:
                active_rntis.append(rnti)
                self.priority_manager.calculate_and_update_priority(rnti)
    
    def _register_ue(self, rnti_str: str, request_size: int, slo_ms: int) -> None:
        """Register a new UE with auto-detected parameters.
        
        Args:
            rnti_str: RNTI as hex string.
            request_size: Size of the first request in bytes.
            slo_ms: Service Level Objective in milliseconds.
        """
        self.app_handler.ue_info[rnti_str] = {
            "app_id": "udp_app",
            "ue_idx": "auto",
            "latency_req": float(slo_ms),
            "request_size": request_size,
        }
        self.app_handler.ue_resource_needs[rnti_str] = 0
        self.app_handler.ue_pending_requests[rnti_str] = {}
        self.app_handler.request_start_times[rnti_str] = {}
        
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
        # Clean up request start times
        if rnti_str in self.app_handler.request_start_times:
            expired_req_ids = [
                req_id for req_id in self.app_handler.request_start_times[rnti_str] 
                if req_id <= request_index
            ]
            for req_id in expired_req_ids:
                if req_id in self.app_handler.request_start_times[rnti_str]:
                    del self.app_handler.request_start_times[rnti_str][req_id]
        
        # Clean up PRB allocations
        self.metrics_processor.cleanup_request_prb_allocations(rnti_str, request_index)