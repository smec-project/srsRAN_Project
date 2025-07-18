"""Priority management functionality for PMEC Controller."""

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
        
        # Track UE priority states
        self.ue_priorities: Dict[str, float] = {}
        self.ue_latest_bsr: Dict[str, int] = {}
        
        # Track request remaining times for each UE
        self.ue_remaining_times: Dict[str, List[Tuple[int, float]]] = {}
        self.ue_last_event_time: Dict[str, float] = {}
        
        # Counter for positive predictions
        self.positive_predictions: Dict[str, int] = {}
        
        # UE information storage
        self.ue_info: Dict[str, dict] = {}
    
    def initialize_ue_priority(self, rnti: str) -> None:
        """Initialize priority tracking for a new UE.
        
        Args:
            rnti: Radio Network Temporary Identifier.
        """
        if rnti not in self.ue_priorities:
            self.ue_priorities[rnti] = 0.0
            self.logger.log(f"Initialized priority for RNTI {rnti}")
    
    def register_ue(
        self, 
        rnti: str, 
        slo_latency: float
    ) -> None:
        """Register a new UE with its requirements.
        
        Args:
            rnti: Radio Network Temporary Identifier.
            slo_latency: SLO latency requirement in milliseconds.
        """
        self.ue_info[rnti] = {
            "slo_latency": slo_latency,
        }
        
        # Initialize tracking structures
        if rnti not in self.ue_remaining_times:
            self.ue_remaining_times[rnti] = []
        if rnti not in self.positive_predictions:
            self.positive_predictions[rnti] = 0
        if rnti not in self.ue_last_event_time:
            self.ue_last_event_time[rnti] = get_current_timestamp()
        
        self.initialize_ue_priority(rnti)
        
        self.logger.log(
            f"Registered UE - RNTI: {rnti}, "
            f"SLO Latency: {slo_latency}ms"
        )
    
    def update_bsr_state(self, rnti: str, bsr_bytes: int) -> None:
        """Update the latest BSR state for a UE.
        
        Args:
            rnti: Radio Network Temporary Identifier.
            bsr_bytes: BSR value in bytes.
        """
        self.ue_latest_bsr[rnti] = bsr_bytes
        
        # If BSR is 0, clear all remaining times
        if bsr_bytes == 0 and rnti in self.ue_remaining_times:
            self.ue_remaining_times[rnti] = []
            self.logger.log(f"Cleared remaining times for RNTI {rnti} (BSR=0)")
    
    def update_remaining_times(self, rnti: str, current_time: float) -> None:
        """Update remaining times for all active requests of a UE.
        
        Args:
            rnti: Radio Network Temporary Identifier.
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
            for req_id, remaining in self.ue_remaining_times[rnti]:
                new_remaining = remaining - time_passed
                if new_remaining > 0:  # Only keep requests with remaining time > 0
                    updated_times.append((req_id, new_remaining))
            
            self.ue_remaining_times[rnti] = updated_times
        
        # Update last event time
        self.ue_last_event_time[rnti] = current_time
    
    def add_new_request(
        self, 
        rnti: str, 
        remaining_time: float, 
        gnb_max_prb_slot: int,
        latest_bsr_slot: int,
        event_processor_data: dict = None
    ) -> None:
        """Add a new request for a UE with calculated remaining time.
        
        Args:
            rnti: Radio Network Temporary Identifier.
            remaining_time: Initial remaining time for the request.
            gnb_max_prb_slot: Global maximum PRB slot.
            latest_bsr_slot: Slot number of the latest BSR.
            event_processor_data: Additional data from event processor.
        """
        if rnti not in self.ue_info:
            self.logger.log(f"Warning: Adding request for unknown RNTI {rnti}")
            return
        
        # Update positive prediction counter
        self.positive_predictions[rnti] += 1
        
        # Calculate adjusted remaining time based on slot differences
        adjusted_remaining = remaining_time - (gnb_max_prb_slot - latest_bsr_slot) * 0.5
        
        # Add new request with its remaining time
        if rnti not in self.ue_remaining_times:
            self.ue_remaining_times[rnti] = []
        
        self.ue_remaining_times[rnti].append(
            (self.positive_predictions[rnti], adjusted_remaining)
        )
        
        self.logger.log(
            f"Added new request for RNTI {rnti}: Request_{self.positive_predictions[rnti]}, "
            f"Remaining time: {adjusted_remaining:.2f}ms"
        )
    
    def calculate_priority(self, rnti: str) -> float:
        """Calculate priority for a UE based on oldest request's remaining time.
        
        Args:
            rnti: Radio Network Temporary Identifier.
            
        Returns:
            Calculated priority value.
        """
        if (rnti not in self.ue_remaining_times or 
            not self.ue_remaining_times[rnti]):
            return 0.0
        
        # Get first (oldest) request's remaining time
        oldest_req = self.ue_remaining_times[rnti][0]
        current_remaining = oldest_req[1]
        
        # Convert remaining time from ms to s and calculate priority
        remaining_seconds = current_remaining / 1000.0
        current_bsr = self.ue_latest_bsr.get(rnti, 0)
        
        # Priority calculation: BSR / (remaining_time^2 + epsilon)
        priority = safe_divide(
            current_bsr, 
            remaining_seconds * remaining_seconds + 1e-6,
            default=0.0
        )
        
        return priority
    
    def get_priority_update_info(self, rnti: str) -> Tuple[bool, float, str]:
        """Get priority update information for a UE.
        
        Args:
            rnti: Radio Network Temporary Identifier.
            
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
                    f"Updated priority for RNTI {rnti}: "
                    f"Priority={new_priority:.2f}, "
                    f"Remaining_time={oldest_req[1]:.2f}ms"
                )
            else:
                log_msg = f"Reset priority for RNTI {rnti}"
        else:
            log_msg = ""
        
        return should_update, new_priority, log_msg
    
    def update_priority(self, rnti: str, new_priority: float) -> None:
        """Update the stored priority for a UE.
        
        Args:
            rnti: Radio Network Temporary Identifier.
            new_priority: New priority value.
        """
        self.ue_priorities[rnti] = new_priority
    
    def get_remaining_times_summary(self, rnti: str) -> str:
        """Get a summary of remaining times for logging.
        
        Args:
            rnti: Radio Network Temporary Identifier.
            
        Returns:
            Formatted string with remaining times information.
        """
        if rnti not in self.ue_remaining_times or not self.ue_remaining_times[rnti]:
            return "No active requests"
        
        summary_parts = []
        for req_id, remaining in self.ue_remaining_times[rnti]:
            summary_parts.append(f"Request_{req_id}_remaining={remaining:.2f}ms")
        
        total_positive = self.positive_predictions.get(rnti, 0)
        return f"Total positive predictions: {total_positive}, " + ", ".join(summary_parts)
    
    def cleanup_ue(self, rnti: str) -> None:
        """Clean up all tracking data for a UE.
        
        Args:
            rnti: Radio Network Temporary Identifier.
        """
        tracking_dicts = [
            self.ue_priorities,
            self.ue_latest_bsr,
            self.ue_remaining_times,
            self.ue_last_event_time,
            self.positive_predictions,
            self.ue_info
        ]
        
        for tracking_dict in tracking_dicts:
            if rnti in tracking_dict:
                del tracking_dict[rnti]
        
        self.logger.log(f"Cleaned up tracking data for RNTI {rnti}") 