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
        
        incentive_threshold = latency_req / 2
        
        # Calculate priority based on current state
        if elapsed_time_ms < incentive_threshold:
            priority = self._calculate_incentive_priority(rnti_str)
        elif elapsed_time_ms < latency_req:
            priority = self._calculate_accelerate_priority(rnti_str, earliest_req_id)
        else:
            priority = 100  # High priority for overdue requests
            self.logger.write(
                f"rnti: {rnti_str} latency_req: {latency_req} elapsed_time_ms: "
                f"{elapsed_time_ms} priority: {priority}\n"
            )
            self.logger.flush()
        
        # Update priority state and send to scheduler
        self.ue_priorities[rnti_str] = priority
        self.set_priority(rnti_str, priority)
        
        self.logger.write(f"PRIORITY_SET: RNTI {rnti_str} -> priority {priority}\n")
        self.logger.flush()
    
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
    
    def _calculate_ue_prb_requirements(self) -> Dict[str, Tuple[int, int]]:
        """Calculate PRB requirements for all UEs with active requests.
        
        Returns:
            Dictionary mapping rnti -> (total_prbs, prbs_per_tti).
        """
        ue_prb_requirements = {}
        
        # Calculate for each UE with active requests
        all_rntis = list(self.app_handler.request_start_times.keys())
        for rnti in all_rntis:
            request_times = self.app_handler.get_request_start_times(rnti)
            if not request_times:
                continue
            
            # Get earliest request
            earliest_req_id = min(request_times.items(), key=lambda x: x[1])[0]
            pending_requests = self.app_handler.get_pending_requests(rnti)
            
            if earliest_req_id not in pending_requests:
                continue
                
            request_size = pending_requests[earliest_req_id]
            total_prbs = calculate_prbs_needed(request_size, self.config.bytes_per_prb)
            
            ue_info = self.app_handler.get_ue_info(rnti)
            if not ue_info:
                continue
                
            latency_req = ue_info['latency_req']
            available_ttis = latency_req / self.config.tti_duration_ms
            prbs_per_tti = max(1, (total_prbs + int(available_ttis) - 1) // int(available_ttis))
            
            ue_prb_requirements[rnti] = (total_prbs, prbs_per_tti)
        
        return ue_prb_requirements
    
    def _calculate_incentive_priority(self, ue_rnti: str) -> float:
        """Calculate priority for incentive mode (first half of latency requirement).
        
        This algorithm adjusts priority based on the difference between required
        and actual PRB allocations, implementing a game-theoretic approach.
        
        Args:
            ue_rnti: RNTI as hex string.
            
        Returns:
            Calculated priority value.
        """
        # Get UE requirements and calculate PRB needs
        ue_prb_requirements = self._calculate_ue_prb_requirements()
        
        # Get the latest slot from history
        latest_slot = self.metrics_processor.get_latest_slot()
        if not latest_slot or ue_rnti not in ue_prb_requirements:
            return self.config.default_priority_offset
        
        # Calculate priority adjustments based on PRB differences
        priority_adjustments = {}
        for rnti, (_, required_prbs_per_tti) in ue_prb_requirements.items():
            prb_history = self.metrics_processor.get_prb_history(rnti)
            if not prb_history or latest_slot not in prb_history:
                self.logger.write(f"Warning: No PRB history for RNTI {rnti}\n")
                self.logger.flush()
                continue
            
            actual_prbs = prb_history[latest_slot]
            prb_difference = actual_prbs - required_prbs_per_tti
            current_offset = self.ue_priorities.get(rnti, self.config.default_priority_offset)
            priority_adjustments[rnti] = prb_difference * current_offset
        
        if not priority_adjustments or ue_rnti not in priority_adjustments:
            return self.config.default_priority_offset
        
        # Apply incentive algorithm logic
        total_priority_metric = sum(priority_adjustments.values())
        ue_priority_metric = priority_adjustments[ue_rnti]
        current_offset = self.ue_priorities.get(ue_rnti, self.config.default_priority_offset)
        
        prb_history = self.metrics_processor.get_prb_history(ue_rnti)
        actual_prbs = prb_history[latest_slot] if prb_history and latest_slot in prb_history else 0
        required_prbs = ue_prb_requirements[ue_rnti][1]
        
        if ue_priority_metric > 0 and total_priority_metric < 0:
            current_offset = current_offset / 2
        else:
            if actual_prbs > 0:
                current_offset = self.ue_priorities.get(ue_rnti, self.config.default_priority_offset) + max(
                    self.config.default_priority_offset, 
                    abs(actual_prbs - required_prbs) / actual_prbs
                )
            else:
                current_offset = self.ue_priorities.get(ue_rnti, self.config.default_priority_offset) + abs(actual_prbs - required_prbs)
        
        self.logger.write(
            f"incentive {ue_rnti} {actual_prbs} {required_prbs} {current_offset}\n"
        )
        self.logger.flush()
        return current_offset
    
    def _calculate_accelerate_priority(self, ue_rnti: str, request_id: int) -> float:
        """Calculate priority for accelerate mode (second half of latency requirement).
        
        This algorithm uses exponential decay based on time to deadline and
        remaining PRB requirements.
        
        Args:
            ue_rnti: RNTI as hex string.
            request_id: ID of the request.
            
        Returns:
            Calculated priority value.
        """
        # Get current request info
        ue_info = self.app_handler.get_ue_info(ue_rnti)
        pending_requests = self.app_handler.get_pending_requests(ue_rnti)
        request_start_times = self.app_handler.get_request_start_times(ue_rnti)
        
        if not ue_info or request_id not in pending_requests or request_id not in request_start_times:
            return 0.0
        
        # Calculate timing information
        start_time = request_start_times[request_id]
        elapsed_time_ms = (time.time() - start_time) * 1000
        latency_req_ms = ue_info['latency_req']
        time_to_deadline_s = (latency_req_ms - elapsed_time_ms) * self.config.ms_to_seconds
        
        # Calculate remaining PRBs needed
        request_size = pending_requests[request_id]
        total_prbs_needed = calculate_prbs_needed(request_size, self.config.bytes_per_prb)
        
        # Get allocated PRBs for this request
        prbs_allocated = self.metrics_processor.get_request_prb_allocation(ue_rnti, request_id)
        
        if total_prbs_needed > prbs_allocated:
            remaining_prbs = total_prbs_needed - prbs_allocated
        else:
            remaining_prbs = 50  # Default value when request is satisfied
        
        # Calculate priority using exponential decay
        priority = remaining_prbs * math.exp(-1 * time_to_deadline_s)
        
        self.logger.write(
            f"accelerate {ue_rnti} {request_id} {prbs_allocated} {total_prbs_needed} "
            f"{time_to_deadline_s} {priority}\n"
        )
        self.logger.flush()
        return priority
    
    def _calculate_ue_prb_requirements(self) -> Dict[str, Tuple[int, int]]:
        """Calculate PRB requirements for all UEs with active requests.
        
        Returns:
            Dictionary mapping RNTI -> (total_prbs, prbs_per_tti).
        """
        ue_prb_requirements = {}
        
        # Get all UEs with active requests
        for rnti_str in self.app_handler.request_start_times.keys():
            if not self.app_handler.request_start_times[rnti_str]:
                continue
            
            # Get earliest request
            earliest_req_id = min(
                self.app_handler.request_start_times[rnti_str].items(), 
                key=lambda x: x[1]
            )[0]
            
            pending_requests = self.app_handler.get_pending_requests(rnti_str)
            ue_info = self.app_handler.get_ue_info(rnti_str)
            
            if earliest_req_id not in pending_requests or not ue_info:
                continue
            
            request_size = pending_requests[earliest_req_id]
            total_prbs = calculate_prbs_needed(request_size, self.config.bytes_per_prb)
            latency_req = ue_info['latency_req']
            available_ttis = latency_req / self.config.tti_duration_ms
            prbs_per_tti = (total_prbs + int(available_ttis) - 1) // int(available_ttis)
            
            ue_prb_requirements[rnti_str] = (total_prbs, prbs_per_tti)
        
        return ue_prb_requirements
    
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