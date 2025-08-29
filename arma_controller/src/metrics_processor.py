"""RAN metrics processor for Tutti Controller."""

import struct
from typing import Dict, List, Optional

from .config import ControllerConfig, DefaultValues
from .utils import Logger, rnti_to_hex_string
from .network_handler import NetworkHandler


class MetricsProcessor:
    """Processes RAN metrics including PRB allocations, SR, and BSR data.
    
    This class handles incoming RAN metrics messages and maintains historical
    data for PRB allocations across time slots.
    """
    
    def __init__(self, config: ControllerConfig, logger: Logger, network_handler: NetworkHandler):
        """Initialize the metrics processor.
        
        Args:
            config: Controller configuration settings.
            logger: Logger instance for writing events.
            network_handler: Network handler for receiving metrics.
        """
        self.config = config
        self.logger = logger
        self.network_handler = network_handler
        
        # Metrics tracking data structures
        self.current_metrics: Dict[str, dict] = {}  # RNTI (str) -> metrics
        self.ue_prb_status: Dict[str, tuple] = {}  # RNTI -> (slot, prbs)
        
        # PRB allocation history
        self.prb_history: Dict[str, Dict[int, int]] = {}  # RNTI -> {slot -> prbs}
        self.slot_history: List[int] = []  # Ordered list of slots
        self.request_prb_allocations: Dict[str, Dict[int, int]] = {}  # RNTI -> {request_id -> total_prbs}
        
        # Current slot tracking for batched updates
        self.current_slot: Optional[int] = None
        self.slot_prb_allocations: Dict[str, int] = {}  # Track RNTIs that got PRBs in current slot
    
    def process_ran_metrics(self) -> None:
        """Process incoming RAN metrics messages continuously."""
        while True:  # This will be controlled by the main controller's running flag
            message_data = self.network_handler.receive_ran_metrics()
            if message_data:
                data, _ = message_data
                self._process_metrics_data(data)
    
    def _process_metrics_data(self, data: bytes) -> None:
        """Process binary RAN metrics data.
        
        Args:
            data: Raw binary metrics data.
        """
        if not data:
            return
        
        # Process binary messages (16 bytes each: 4 x 32-bit integers)
        message_size = DefaultValues.METRICS_MESSAGE_SIZE
        offset = 0
        
        while offset + message_size <= len(data):
            # Unpack: msg_type, rnti, field1, field2
            msg_type, rnti, field1, field2 = struct.unpack(
                'IIII', data[offset:offset + message_size]
            )
            
            if msg_type == 0:  # PRB allocation message
                self._handle_prb_metrics(rnti, field1, field2)
            elif msg_type == 1:  # Scheduling Request (SR) message
                self._handle_sr_metrics(rnti, field1)
            elif msg_type == 2:  # Buffer Status Report (BSR) message
                self._handle_bsr_metrics(rnti, field1, field2)
            else:
                self.logger.write(f"Unknown metrics type: {msg_type}\n")
                self.logger.flush()
            
            offset += message_size
    
    def _handle_prb_metrics(self, rnti: int, prbs: int, slot: int) -> None:
        """Handle PRB allocation metrics.
        
        Args:
            rnti: RNTI as integer value.
            prbs: Number of PRBs allocated.
            slot: Slot number for the allocation.
        """
        # Convert RNTI to string for compatibility
        rnti_str = rnti_to_hex_string(rnti)
        
        # Store basic metrics
        self.current_metrics[rnti_str] = {
            "PRBs": prbs,
            "SLOT": slot,
        }
        
        # Update latest PRB allocation and slot
        self.ue_prb_status[rnti_str] = (slot, prbs)
        
        # Handle slot transitions and return whether slot changed
        slot_changed = self._handle_slot_transition(slot)
        
        # Update PRB history for the current RNTI
        self._update_prb_history(rnti_str, slot, prbs)
        
        # Track which RNTIs got PRBs in this slot
        if prbs > 0:
            self.slot_prb_allocations[rnti_str] = prbs
            
        # Return slot change info for controller to handle priority updates
        return slot_changed
    
    def _handle_sr_metrics(self, rnti: int, slot: int) -> None:
        """Handle Scheduling Request (SR) metrics.
        
        Args:
            rnti: RNTI as integer value (unused currently).
            slot: Slot number for the SR (unused currently).
        """
        # SR metrics are currently not processed
        # This can be extended for future SR-based scheduling logic
        pass
    
    def _handle_bsr_metrics(self, rnti: int, bytes_val: int, slot: int) -> None:
        """Handle Buffer Status Report (BSR) metrics.
        
        Args:
            rnti: RNTI as integer value.
            bytes_val: Number of bytes reported in BSR.
            slot: Slot number for the BSR.
        """
        # Convert RNTI to string for compatibility
        rnti_str = rnti_to_hex_string(rnti)
        
        if rnti_str not in self.current_metrics:
            self.current_metrics[rnti_str] = {}
        
        self.current_metrics[rnti_str].update({
            "BSR_BYTES": bytes_val,
            "BSR_SLOT": slot
        })
    
    def _handle_slot_transition(self, new_slot: int) -> bool:
        """Handle transition to a new slot.
        
        Args:
            new_slot: The new slot number.
        
        Returns:
            True if slot transition occurred, False otherwise.
        """
        slot_changed = False
        
        # Check if we moved to a new slot
        if self.current_slot is not None and new_slot != self.current_slot:
            slot_changed = True
            # Process previous slot: update RNTIs that didn't get PRBs to 0
            self._finalize_previous_slot()
            
            # Reset for new slot
            self.slot_prb_allocations = {}
        
        # Update current slot
        self.current_slot = new_slot
        return slot_changed
    
    def _finalize_previous_slot(self) -> None:
        """Finalize PRB allocations for the previous slot."""
        # Reset slot tracking for new slot
        pass
    
    def _update_prb_history(self, rnti_str: str, slot: int, prbs: int) -> None:
        """Update PRB history and track allocations for active requests.
        
        Args:
            rnti_str: RNTI as hex string.
            slot: Slot number.
            prbs: Number of PRBs allocated.
        """
        # Initialize history for new UE if needed
        if rnti_str not in self.prb_history:
            self.prb_history[rnti_str] = {}
            # Fill with zeros for all known slots
            for s in self.slot_history:
                self.prb_history[rnti_str][s] = 0
        
        # Handle new slot
        if slot not in self.slot_history:
            self.slot_history.append(slot)
            
            # Add zero entries for this slot for all UEs
            for ue_rnti in self.prb_history:
                self.prb_history[ue_rnti][slot] = 0
            
            # Remove oldest slots if beyond window size
            while len(self.slot_history) > self.config.history_window:
                oldest_slot = self.slot_history.pop(0)
                # Remove this slot from all UE histories
                for ue_rnti in self.prb_history:
                    self.prb_history[ue_rnti].pop(oldest_slot, None)
        
        # Update the PRB allocation for this UE and slot
        self.prb_history[rnti_str][slot] = prbs
    
    def update_request_prb_allocation(
        self, 
        rnti_str: str, 
        request_id: int, 
        slot: int, 
        prbs: int
    ) -> None:
        """Update PRB allocation for a specific request.
        
        Args:
            rnti_str: RNTI as hex string.
            request_id: ID of the request.
            slot: Slot number.
            prbs: Number of PRBs allocated.
        """
        # Initialize allocation tracking if needed
        if rnti_str not in self.request_prb_allocations:
            self.request_prb_allocations[rnti_str] = {}
        
        # Initialize or update PRB allocation for this request
        if request_id not in self.request_prb_allocations[rnti_str]:
            self.request_prb_allocations[rnti_str][request_id] = 0
        self.request_prb_allocations[rnti_str][request_id] += prbs
        
        if prbs > 0:
            self.logger.write(
                f"PRB_event - RNTI: {rnti_str}, req_id: {request_id}, prbs: {prbs}, "
                f"total: {self.request_prb_allocations[rnti_str][request_id]}, slot: {slot}\n"
            )
            self.logger.flush()
    
    def cleanup_request_prb_allocations(self, rnti_str: str, request_index: int) -> None:
        """Clean up PRB allocations for completed requests.
        
        Args:
            rnti_str: RNTI as hex string.
            request_index: Index of the completed request.
        """
        if rnti_str in self.request_prb_allocations:
            expired_req_ids = [
                req_id for req_id in self.request_prb_allocations[rnti_str] 
                if req_id <= request_index
            ]
            for req_id in expired_req_ids:
                if req_id in self.request_prb_allocations[rnti_str]:
                    del self.request_prb_allocations[rnti_str][req_id]
    
    def get_prb_history(self, rnti_str: str) -> Dict[int, int]:
        """Get PRB allocation history for a UE.
        
        Args:
            rnti_str: RNTI as hex string.
            
        Returns:
            Dictionary mapping slot -> PRBs allocated.
        """
        return self.prb_history.get(rnti_str, {})
    
    def get_latest_slot(self) -> Optional[int]:
        """Get the latest slot number from history.
        
        Returns:
            Latest slot number or None if no history.
        """
        return self.slot_history[-1] if self.slot_history else None
    
    def get_request_prb_allocation(self, rnti_str: str, request_id: int) -> int:
        """Get total PRB allocation for a specific request.
        
        Args:
            rnti_str: RNTI as hex string.
            request_id: ID of the request.
            
        Returns:
            Total PRBs allocated for the request.
        """
        if (rnti_str in self.request_prb_allocations and 
            request_id in self.request_prb_allocations[rnti_str]):
            return self.request_prb_allocations[rnti_str][request_id]
        return 0
    
    def get_rntis_with_requests(self) -> List[str]:
        """Get list of RNTIs that have active requests.
        
        Returns:
            List of RNTI strings.
        """
        return list(self.request_start_times.keys()) if hasattr(self, 'request_start_times') else []