"""Priority management for Tutti Controller."""

import math
import time
from typing import Dict, Optional, List, Tuple

from .config import ControllerConfig
from .utils import Logger, calculate_prbs_needed
from .network_handler import NetworkHandler
from .app_handler import AppHandler
from .metrics_processor import MetricsProcessor


class PriorityManager:
    """Manages priority calculations and updates for UE traffic scheduling.
    
    This class implements priority calculation algorithms including incentive
    and accelerate modes based on latency requirements and PRB allocations.
    """
    
    def __init__(
        self, 
        config: ControllerConfig, 
        logger: Logger, 
        network_handler: NetworkHandler,
        app_handler: AppHandler,
        metrics_processor: MetricsProcessor
    ):
        """Initialize the priority manager.
        
        Args:
            config: Controller configuration settings.
            logger: Logger instance for writing events.
            network_handler: Network handler for sending priority updates.
            app_handler: Application handler for UE and request information.
            metrics_processor: Metrics processor for PRB history.
        """
        self.config = config
        self.logger = logger
        self.network_handler = network_handler
        self.app_handler = app_handler
        self.metrics_processor = metrics_processor
        
        # Priority state tracking
        self.ue_priorities: Dict[str, float] = {}  # RNTI -> current_priority
    
    def calculate_and_update_priority(self, rnti_str: str) -> None:
        """Calculate and update priority for a specific UE.
        
        Args:
            rnti_str: RNTI as hex string.
        """
        # Skip if we don't have UE info yet
        ue_info = self.app_handler.get_ue_info(rnti_str)
        if not ue_info:
            return
        
        # Initialize priority if not exists
        if rnti_str not in self.ue_priorities:
            self._initialize_ue_priority(rnti_str)
        
        # Skip priority adjustment for requests with latency requirement > threshold
        latency_req = ue_info['latency_req']
        if latency_req > self.config.max_latency_threshold_ms:
            return
        
        requests = self.app_handler.get_request_start_times(rnti_str)
        if not requests:  # No requests for this UE
            self.reset_priority(rnti_str)
            return
        
        # Find earliest valid request
        valid_requests = self._get_valid_requests(rnti_str, requests)
        if not valid_requests:
            # All current requests have been completed, reset priority
            self.reset_priority(rnti_str)
            return
        
        # Calculate priority based on request age
        earliest_req_id, start_time = min(valid_requests.items(), key=lambda x: x[1])
        current_time = time.time()
        elapsed_time_ms = (current_time - start_time) * 1000
        
        # Calculate priority using new formula: frame_size / (prb * remaining_time)
        priority = self._calculate_priority_with_formula(rnti_str, earliest_req_id, elapsed_time_ms, latency_req)
        
        # Update priority state and send to scheduler
        self.ue_priorities[rnti_str] = priority
        self.set_priority(rnti_str, priority)
        
        self.logger.write(f"PRIORITY_SET: RNTI {rnti_str} -> priority {priority}\n")
        self.logger.flush()
    
    def _calculate_priority_with_formula(
        self, rnti_str: str, earliest_req_id: int, elapsed_time_ms: float, latency_req: float
    ) -> float:
        """Calculate priority using formula: frame_size / (prb * remaining_time).
        
        Args:
            rnti_str: RNTI as hex string.
            earliest_req_id: ID of the earliest request.
            elapsed_time_ms: Elapsed time since request start in milliseconds.
            latency_req: Latency requirement in milliseconds.
            
        Returns:
            Calculated priority value.
        """
        # Get request size (frame_size)
        pending_requests = self.app_handler.get_pending_requests(rnti_str)
        if earliest_req_id not in pending_requests:
            return 0.0
        
        frame_size = pending_requests[earliest_req_id]
        
        # Get total PRBs allocated for this request
        prbs_allocated = self.metrics_processor.get_request_prb_allocation(rnti_str, earliest_req_id)
        if prbs_allocated == 0:
            prbs_allocated = 1  # Avoid division by zero
        
        # Calculate remaining_time in seconds
        remaining_time_ms = latency_req - elapsed_time_ms
        remaining_time_s = remaining_time_ms / 1000.0
        if remaining_time_s <= 0:
            remaining_time_s = 0.01  # Use 0.01s if <= 0
        
        # Calculate priority using the formula
        priority = frame_size / (prbs_allocated * remaining_time_s)
        
        self.logger.write(
            f"PRIORITY_CALC: RNTI {rnti_str} req_id {earliest_req_id} "
            f"frame_size: {frame_size} prbs: {prbs_allocated} remaining_time: {remaining_time_s:.3f}s "
            f"priority: {priority:.6f}\n"
        )
        self.logger.flush()
        
        return priority
    
    def _get_valid_requests(self, rnti_str: str, requests: Dict[int, float]) -> Dict[int, float]:
        """Get requests that haven't been completed yet.
        
        Args:
            rnti_str: RNTI as hex string.
            requests: Dictionary of request_id -> start_time.
            
        Returns:
            Dictionary of valid (uncompleted) requests.
        """
        last_completed = self.app_handler.get_last_completed_request_id(rnti_str)
        return {
            req_id: timestamp for req_id, timestamp in requests.items() 
            if req_id > last_completed
        }
    
    
    def set_priority(self, rnti_str: str, priority: float) -> bool:
        """Send priority update to RAN via network handler.
        
        Args:
            rnti_str: RNTI as hex string.
            priority: Priority value to set.
            
        Returns:
            True if update was sent successfully, False otherwise.
        """
        try:
            # Convert string RNTI to integer for binary format
            rnti_int = int(rnti_str, 16)
            return self.network_handler.send_priority_update(rnti_int, priority, False)
        except ValueError:
            self.logger.write(f"Invalid RNTI format: {rnti_str}\n")
            self.logger.flush()
            return False
    
    def reset_priority(self, rnti_str: str) -> bool:
        """Reset priority for a specific RNTI.
        
        Args:
            rnti_str: RNTI as hex string.
            
        Returns:
            True if reset was sent successfully, False otherwise.
        """
        try:
            # Reset priority state
            if rnti_str in self.ue_priorities:
                self.ue_priorities[rnti_str] = 0.0
            
            # Convert string RNTI to integer for binary format
            rnti_int = int(rnti_str, 16)
            return self.network_handler.send_priority_update(rnti_int, 0.0, True)
        except ValueError:
            self.logger.write(f"Invalid RNTI format: {rnti_str}\n")
            self.logger.flush()
            return False
    
    def _initialize_ue_priority(self, rnti_str: str) -> None:
        """Initialize priority for a new UE.
        
        Args:
            rnti_str: RNTI as hex string.
        """
        self.ue_priorities[rnti_str] = 0.0
    
    def process_slot_completion(self, rnti_str: str) -> None:
        """Process completion of a slot for priority updates.
        
        Args:
            rnti_str: RNTI as hex string.
        """
        # This method is called when a slot is completed and priority updates are needed
        self.calculate_and_update_priority(rnti_str)
    
    def get_current_priority(self, rnti_str: str) -> float:
        """Get current priority for a UE.
        
        Args:
            rnti_str: RNTI as hex string.
            
        Returns:
            Current priority value.
        """
        return self.ue_priorities.get(rnti_str, 0.0)