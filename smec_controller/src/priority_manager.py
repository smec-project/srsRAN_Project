"""Priority management functionality for SMEC Controller."""


from typing import Dict, List, Tuple

from .config import ControllerConfig
from .utils import Logger, get_current_timestamp, safe_divide


class PriorityManager:
    """Manages UE priorities and request tracking.
    
    Handles priority calculations based on remaining request times,
    tracks active requests per UE, and manages priority updates.
    """
    
    def __init__(self, config: ControllerConfig, logger: Logger):
        """Initialize the priority manager.
        
        Args:
            config: Configuration settings.
            logger: Logger instance for debugging output.
        """
        self.config = config
        self.logger = logger
        
        # Track UE priority states - using int RNTI as keys
        self.ue_priorities: Dict[int, float] = {}
        self.ue_latest_bsr: Dict[int, int] = {}
        
        # Track request remaining times and sizes for each UE
        self.ue_remaining_times: Dict[int, List[Tuple[int, float, int]]] = {}
        self.ue_last_event_time: Dict[int, float] = {}
        
        # Track BSR decrease accumulation for intelligent request removal
        self.ue_bsr_decrease_accumulation: Dict[int, int] = {}
        
        # Counter for positive predictions
        self.positive_predictions: Dict[int, int] = {}
        
        # UE information storage
        self.ue_info: Dict[int, dict] = {}
        
        # Global sequence number tracking
        self.global_sequence_counter: int = 0
        self.rnti_to_sequence: Dict[int, int] = {}
    
    def initialize_ue_priority(self, rnti: int) -> None:
        """Initialize priority tracking for a new UE.
        
        Args:
            rnti: Radio Network Temporary Identifier as integer.
        """
        if rnti not in self.ue_priorities:
            self.ue_priorities[rnti] = 0.0
            self.logger.log(f"Initialized priority for RNTI 0x{rnti:x}")
    
    def register_ue(
        self, 
        rnti: int, 
        slo_latency: float
    ) -> None:
        """Register a new UE with its requirements.
        
        Args:
            rnti: Radio Network Temporary Identifier as integer.
            slo_latency: SLO latency requirement in milliseconds.
        """
        # Assign global sequence number if UE is not already registered
        if rnti not in self.rnti_to_sequence:
            self.global_sequence_counter += 1
            self.rnti_to_sequence[rnti] = self.global_sequence_counter
        
        sequence_number = self.rnti_to_sequence[rnti]
        
        self.ue_info[rnti] = {
            "slo_latency": slo_latency,
            "sequence_number": sequence_number,
        }
        
        # Initialize tracking structures
        if rnti not in self.ue_remaining_times:
            self.ue_remaining_times[rnti] = []
        if rnti not in self.ue_bsr_decrease_accumulation:
            self.ue_bsr_decrease_accumulation[rnti] = 0
        if rnti not in self.positive_predictions:
            self.positive_predictions[rnti] = 0
        if rnti not in self.ue_last_event_time:
            self.ue_last_event_time[rnti] = get_current_timestamp()
        
        self.initialize_ue_priority(rnti)
        
        self.logger.log(
            f"Registered UE - RNTI: 0x{rnti:x}, "
            f"Sequence: {sequence_number}, "
            f"SLO Latency: {slo_latency}ms"
        )
    
    def update_bsr_state(self, rnti: int, bsr_bytes: int) -> None:
        """Update the latest BSR state for a UE and handle request removal logic.
        
        Args:
            rnti: Radio Network Temporary Identifier as integer.
            bsr_bytes: BSR value in bytes.
        """
        # Get previous BSR value
        previous_bsr = self.ue_latest_bsr.get(rnti, 0)
        
        # Update current BSR
        self.ue_latest_bsr[rnti] = bsr_bytes
        
        # Initialize accumulation if not exists
        if rnti not in self.ue_bsr_decrease_accumulation:
            self.ue_bsr_decrease_accumulation[rnti] = 0
        
        # If BSR is 0, clear all remaining times and reset accumulation
        if bsr_bytes == 0:
            if rnti in self.ue_remaining_times:
                self.ue_remaining_times[rnti] = []
            self.ue_bsr_decrease_accumulation[rnti] = 0
            # self.logger.log(f"Cleared remaining times for RNTI 0x{rnti:x} (BSR=0)")
            return
        
        # Calculate BSR change
        bsr_change = bsr_bytes - previous_bsr
        
        # If BSR decreased, accumulate the decrease
        if bsr_change < 0:
            decrease_amount = abs(bsr_change)
            self.ue_bsr_decrease_accumulation[rnti] += decrease_amount
            
            # self.logger.log(
            #     f"BSR decreased for RNTI 0x{rnti:x}: -{decrease_amount} bytes, "
            #     f"Total accumulated: {self.ue_bsr_decrease_accumulation[rnti]} bytes"
            # )
            
            # Check if we can remove the oldest request
            self._check_and_remove_completed_requests(rnti)
    
    def _check_and_remove_completed_requests(self, rnti: int) -> None:
        """Check and remove completed requests based on BSR decrease accumulation.
        
        Args:
            rnti: Radio Network Temporary Identifier as integer.
        """
        if (rnti not in self.ue_remaining_times or 
            not self.ue_remaining_times[rnti] or
            rnti not in self.ue_bsr_decrease_accumulation):
            return
        
        accumulated_decrease = self.ue_bsr_decrease_accumulation[rnti]
        
        # Remove requests while accumulated decrease is sufficient
        while (self.ue_remaining_times[rnti] and 
               accumulated_decrease > 0):
            
            oldest_request = self.ue_remaining_times[rnti][0]
            req_id, remaining_time, request_size = oldest_request
            
            # If accumulated decrease >= request size, remove this request
            if accumulated_decrease >= request_size:
                self.ue_remaining_times[rnti].pop(0)
                accumulated_decrease -= request_size
                
                # self.logger.log(
                #     f"Removed completed request for RNTI 0x{rnti:x}: "
                #     f"Request_{req_id} (size={request_size} bytes), "
                #     f"Remaining accumulated: {accumulated_decrease} bytes"
                # )
            else:
                # Not enough accumulated decrease to remove this request
                break
        
        # Update the remaining accumulation
        self.ue_bsr_decrease_accumulation[rnti] = accumulated_decrease
        
        # If no more requests, reset accumulation to 0
        if not self.ue_remaining_times[rnti]:
            self.ue_bsr_decrease_accumulation[rnti] = 0
    
    def update_all_remaining_times(self, current_time: float) -> None:
        """Update remaining times for all registered UEs.
        
        Args:
            current_time: Current timestamp.
        """
        for rnti in self.ue_info.keys():
            self.update_remaining_times(rnti, current_time)
    
    def update_remaining_times(self, rnti: int, current_time: float) -> None:
        """Update remaining times for all active requests of a UE.
        
        Args:
            rnti: Radio Network Temporary Identifier as integer.
            current_time: Current timestamp.
        """
        if (rnti not in self.ue_remaining_times or 
            not self.ue_remaining_times[rnti] or
            rnti not in self.ue_last_event_time):
            return
        
        time_passed = (current_time - self.ue_last_event_time[rnti]) * 1000  # Convert to ms
        
        if time_passed > 0:
            # Update all remaining times
            updated_times = []
            for req_id, remaining, size in self.ue_remaining_times[rnti]:
                new_remaining = remaining - time_passed
                # Only keep requests with remaining_time >= -10ms
                # if new_remaining >= -10:
                updated_times.append((req_id, new_remaining, size))
            self.ue_remaining_times[rnti] = updated_times
        
        # Update last event time
        self.ue_last_event_time[rnti] = current_time
    
    def add_new_request(
        self, 
        rnti: int, 
        remaining_time: float, 
        gnb_max_prb_slot: int,
        latest_bsr_slot: int,
        bsr_increment: int = 0,
        event_processor_data: dict = None
    ) -> None:
        """Add a new request for a UE with calculated remaining time and size.
        
        Args:
            rnti: Radio Network Temporary Identifier as integer.
            remaining_time: Initial remaining time for the request.
            gnb_max_prb_slot: Global maximum PRB slot.
            latest_bsr_slot: Slot number of the latest BSR.
            bsr_increment: BSR increment (current_bsr - previous_bsr) as request size.
            event_processor_data: Additional data from event processor.
        """
        if rnti not in self.ue_info:
            self.logger.log(f"Warning: Adding request for unknown RNTI 0x{rnti:x}")
            return
        
        # Update positive prediction counter
        self.positive_predictions[rnti] += 1

        # self.logger.log(f"remaining_time: {remaining_time}, gnb_max_prb_slot: {gnb_max_prb_slot}, latest_bsr_slot: {latest_bsr_slot}")
        adjusted_remaining = remaining_time
        # Calculate adjusted remaining time based on slot differences
        # adjusted_remaining = remaining_time - (gnb_max_prb_slot - latest_bsr_slot) * 0.5
        
        # Add new request with its remaining time and size
        if rnti not in self.ue_remaining_times:
            self.ue_remaining_times[rnti] = []
        
        self.ue_remaining_times[rnti].append(
            (self.positive_predictions[rnti], adjusted_remaining, bsr_increment)
        )
        
        # Update last event time when adding new request
        self.ue_last_event_time[rnti] = get_current_timestamp()
        
        sequence_number = self.rnti_to_sequence.get(rnti, 0)
        self.logger.log(
            f"ue{sequence_number} added request {self.positive_predictions[rnti]}, "
            f"Remaining time: {adjusted_remaining:.2f} ms, SLO latency: {self.ue_info[rnti]['slo_latency']:.2f} ms"
        )
    
    def calculate_priority(self, rnti: int) -> float:
        """Calculate priority for a UE based on oldest request's remaining time.
        
        Args:
            rnti: Radio Network Temporary Identifier as integer.
            
        Returns:
            Calculated priority value.
        """
        # Check if UE is registered and get slo_latency
        slo_latency = self.ue_info.get(rnti, {}).get("slo_latency", None)
        if slo_latency is not None and slo_latency > 1000:
            # For non-low-latency applications, always return 0 priority
            return 0.0
        
        if (rnti not in self.ue_remaining_times or 
            not self.ue_remaining_times[rnti]):
            return 0.0
        
        # Get first (oldest) request's remaining time and size
        oldest_req = self.ue_remaining_times[rnti][0]
        current_remaining = oldest_req[1]
        request_size = oldest_req[2]
        
        # Convert remaining time from ms to s and calculate priority
        remaining_seconds = current_remaining / 1000.0
        current_bsr = self.ue_latest_bsr.get(rnti, 0)
        
        if current_remaining < 0:
            # Use new formula when remaining_time < 0
            priority = safe_divide(current_bsr, 1e-8, default=0.0) - current_bsr * current_remaining
        else:
            # Priority calculation: BSR / (remaining_time^2 + epsilon)
            priority = safe_divide(
                current_bsr, 
                remaining_seconds * remaining_seconds + 1e-8,
                default=0.0
            )
        
        return priority
    
    def get_priority_update_info(self, rnti: int) -> Tuple[bool, float, str]:
        """Get priority update information for a UE.
        
        Args:
            rnti: Radio Network Temporary Identifier as integer.
            
        Returns:
            Tuple of (should_update, new_priority, log_message).
        """
        if rnti not in self.ue_info:
            return False, 0.0, ""
        
        # Ensure UE is initialized
        if self.ue_priorities.get(rnti, -1) == -1:
            self.initialize_ue_priority(rnti)
        
        new_priority = self.calculate_priority(rnti)
        current_priority = self.ue_priorities.get(rnti, 0)
        
        # Check if priority should be updated
        should_update = current_priority != new_priority
        
        # Prepare log message
        if should_update:
            if new_priority > 0:
                oldest_req = self.ue_remaining_times[rnti][0]
                log_msg = (
                    f"Updated priority for RNTI 0x{rnti:x}: "
                    f"Priority={new_priority:.2f}, "
                    f"Remaining_time={oldest_req[1]:.2f}ms, "
                    f"Size={oldest_req[2]} bytes"
                )
            else:
                log_msg = f"Reset priority for RNTI 0x{rnti:x}"
        else:
            log_msg = ""
        
        return should_update, new_priority, log_msg
    
    def update_priority(self, rnti: int, new_priority: float) -> None:
        """Update the stored priority for a UE.
        
        Args:
            rnti: Radio Network Temporary Identifier as integer.
            new_priority: New priority value.
        """
        self.ue_priorities[rnti] = new_priority
    
    def get_remaining_times_summary(self, rnti: int) -> str:
        """Get a summary of remaining times for logging.
        
        Args:
            rnti: Radio Network Temporary Identifier as integer.
            
        Returns:
            Formatted string with remaining times information.
        """
        if rnti not in self.ue_remaining_times or not self.ue_remaining_times[rnti]:
            return "No active requests"
        
        summary_parts = []
        for req_id, remaining, size in self.ue_remaining_times[rnti]:
            summary_parts.append(f"Request_{req_id}_remaining={remaining:.2f}ms_size={size}bytes")
        
        total_positive = self.positive_predictions.get(rnti, 0)
        return f"Total positive predictions: {total_positive}, " + ", ".join(summary_parts)
    
    def cleanup_ue(self, rnti: int) -> None:
        """Clean up all tracking data for a UE.
        
        Args:
            rnti: Radio Network Temporary Identifier as integer.
        """
        tracking_dicts = [
            self.ue_priorities,
            self.ue_latest_bsr,
            self.ue_remaining_times,
            self.ue_last_event_time,
            self.ue_bsr_decrease_accumulation,
            self.positive_predictions,
            self.ue_info,
            self.rnti_to_sequence
        ]
        
        for tracking_dict in tracking_dicts:
            if rnti in tracking_dict:
                del tracking_dict[rnti]
        
        self.logger.log(f"Cleaned up tracking data for RNTI 0x{rnti:x}") 