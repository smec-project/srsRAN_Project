"""RAN metrics processing functionality for PMEC Controller."""

import struct
from typing import Dict, List, Optional, Any
from collections import deque

from .config import EventTypes
from .utils import Logger, get_current_timestamp
from .event_processor import EventProcessor
from .priority_manager import PriorityManager
from .model_inference import ModelInference


class MetricsProcessor:
    """Processes RAN metrics and coordinates inference and priority updates.
    
    Handles SR, BSR, and PRB metrics from the RAN and coordinates
    with other components for event processing and priority management.
    """
    
    def __init__(
        self,
        event_processor: EventProcessor,
        priority_manager: PriorityManager,
        model_inference: ModelInference,
        logger: Logger
    ):
        """Initialize the metrics processor.
        
        Args:
            event_processor: Event processor instance.
            priority_manager: Priority manager instance.
            model_inference: Model inference instance.
            logger: Logger instance for debugging output.
        """
        self.event_processor = event_processor
        self.priority_manager = priority_manager
        self.model_inference = model_inference
        self.logger = logger
        
        # Track BSR events for FIFO queue management - using int RNTI as keys
        self.ue_bsr_events: Dict[int, deque] = {}
        self.ue_last_bsr: Dict[int, tuple] = {}
        self.ue_peak_buffer_size: Dict[int, int] = {}
    
    def process_metrics_data(self, data: bytes) -> None:
        """Process incoming RAN metrics binary data.
        
        Args:
            data: Raw binary metrics data from RAN.
        """
        try:
            # Each message is 16 bytes (4 x 32-bit integers)
            message_size = 16
            offset = 0
            
            while offset + message_size <= len(data):
                # Unpack 4 32-bit unsigned integers in native byte order
                msg_type, rnti, field1, field2 = struct.unpack('IIII', data[offset:offset + message_size])
                
                if msg_type == 0:  # PRB
                    self._handle_prb_metrics(rnti, field1, field2)
                elif msg_type == 1:  # SR
                    self._handle_sr_metrics(rnti, field1)
                elif msg_type == 2:  # BSR
                    self._handle_bsr_metrics(rnti, field1, field2)
                else:
                    self.logger.log(f"Unknown metrics type: {msg_type}")
                
                offset += message_size
                    
        except Exception as e:
            self.logger.log(f"Error processing binary metrics data: {e}")
    
    def _handle_sr_metrics(self, rnti: int, slot: int) -> None:
        """Handle SR (Scheduling Request) metrics.
        
        Args:
            rnti: RNTI value as integer.
            slot: Slot number.
        """
        try:
            timestamp = get_current_timestamp()
            
            self.logger.log(
                f"SR received from RNTI=0x{rnti:x}, slot={slot} at {timestamp}"
            )
            
            # Add event to processor - use rnti directly as integer
            self.event_processor.add_event(rnti, "SR", timestamp, slot)
            
        except Exception as e:
            self.logger.log(f"Error processing SR metrics: {e}")
    
    def _handle_bsr_metrics(self, rnti: int, bytes_val: int, slot: int) -> None:
        """Handle BSR (Buffer Status Report) metrics.
        
        Args:
            rnti: RNTI value as integer.
            bytes_val: Buffer size in bytes.
            slot: Slot number.
        """
        try:
            timestamp = get_current_timestamp()
            
            # Update priority manager state - use rnti directly as integer
            self.priority_manager.update_bsr_state(rnti, bytes_val)
            
            # Initialize FIFO queue if not exists
            if rnti not in self.ue_bsr_events:
                self.ue_bsr_events[rnti] = deque()
            
            # Add new BSR event to queue
            self.ue_bsr_events[rnti].append((bytes_val, slot))
            
            # Update peak buffer size tracking
            if rnti not in self.ue_peak_buffer_size:
                self.ue_peak_buffer_size[rnti] = 0
            self.ue_peak_buffer_size[rnti] = max(
                self.ue_peak_buffer_size[rnti], bytes_val
            )
            
            self.logger.log(
                f"BSR received from RNTI=0x{rnti:x}, slot={slot}, "
                f"bytes={bytes_val} at {timestamp}"
            )
            
            # Add event to processor - use rnti directly as integer
            self.event_processor.add_event(rnti, "BSR", timestamp, slot, bytes=bytes_val)
            
            # Process window update and inference
            self._process_bsr_window_update(rnti)
            
        except Exception as e:
            self.logger.log(f"Error processing BSR metrics: {e}")
    
    def _handle_prb_metrics(self, rnti: int, prbs: int, slot: int) -> None:
        """Handle PRB (Physical Resource Block) allocation metrics.
        
        Args:
            rnti: RNTI value as integer.
            prbs: Number of PRBs allocated.
            slot: Slot number.
        """
        try:
            timestamp = get_current_timestamp()
            
            self.logger.log(
                f"PRB received from RNTI=0x{rnti:x}, slot={slot}, "
                f"prbs={prbs} at {timestamp}"
            )
            
            # Add event to processor - use rnti directly as integer
            self.event_processor.add_event(rnti, "PRB", timestamp, slot, prbs=prbs)
            
        except Exception as e:
            self.logger.log(f"Error processing PRB metrics: {e}")
    
    def _process_bsr_window_update(self, rnti: int) -> None:
        """Process window update and inference when BSR arrives.
        
        Args:
            rnti: Radio Network Temporary Identifier as integer.
        """
        try:
            # Trim window and check if ready for analysis
            window_ready = self.event_processor.trim_window(rnti)
            
            if not window_ready:
                return
            
            # Get BSR indices for analysis
            bsr_indices = self.event_processor.get_bsr_indices(rnti)
            if len(bsr_indices) < 2:
                return
            
            # Get events for analysis
            events = self.event_processor.window_events[rnti]
            latest_bsr = events[bsr_indices[-1]]
            prev_bsr = events[bsr_indices[-2]]
            
            # Check if latest BSR bytes increased
            bsr_increased = latest_bsr[1] > prev_bsr[1]
            
            # Perform model inference if available
            if self.model_inference.is_loaded:
                features = self.event_processor.extract_window_features(rnti)
                if features is not None:
                    final_prediction, bsr_inc, model_pred = (
                        self.model_inference.predict_with_bsr_fallback(
                            features, latest_bsr[1], prev_bsr[1]
                        )
                    )
                    
                    # Process prediction result
                    self._process_prediction_result(
                        rnti, final_prediction, bsr_inc, model_pred, 
                        latest_bsr, prev_bsr, bsr_indices
                    )
            else:
                # Fallback to BSR increase only
                self._process_prediction_result(
                    rnti, bsr_increased, bsr_increased, None,
                    latest_bsr, prev_bsr, bsr_indices
                )
                
        except Exception as e:
            self.logger.log(f"Error in BSR window update for RNTI 0x{rnti:x}: {e}")
    
    def _process_prediction_result(
        self,
        rnti: int,
        final_prediction: bool,
        bsr_increased: bool,
        model_prediction: Optional[float],
        latest_bsr,
        prev_bsr,
        bsr_indices: List[int]
    ) -> None:
        """Process the prediction result and update request tracking.
        
        Args:
            rnti: Radio Network Temporary Identifier as integer.
            final_prediction: Final prediction result.
            bsr_increased: Whether BSR increased.
            model_prediction: Model prediction value (can be None).
            latest_bsr: Latest BSR event data.
            prev_bsr: Previous BSR event data.
            bsr_indices: List of BSR indices in the window.
        """
        # Handle zero BSR case
        if latest_bsr[1] == 0:
            # Clear all remaining times
            if rnti in self.priority_manager.ue_remaining_times:
                self.priority_manager.ue_remaining_times[rnti] = []
        else:
            # If prediction is positive, add new request
            if final_prediction:
                self._add_predicted_request(rnti, latest_bsr, prev_bsr, bsr_indices)
        
        # Log prediction and current state
        self._log_prediction_result(
            rnti, final_prediction, bsr_increased, model_prediction
        )
    
    def _add_predicted_request(
        self,
        rnti: int,
        latest_bsr,
        prev_bsr,
        bsr_indices: List[int]
    ) -> None:
        """Add a new predicted request for a UE.
        
        Args:
            rnti: Radio Network Temporary Identifier as integer.
            latest_bsr: Latest BSR event data.
            prev_bsr: Previous BSR event data.
            bsr_indices: List of BSR indices in the window.
        """
        if rnti not in self.priority_manager.ue_info:
            self.logger.log(f"Cannot add request for unknown RNTI 0x{rnti:x}")
            return
        
        # Calculate initial remaining time
        remaining_time = self.priority_manager.ue_info[rnti]["slo_latency"]
        
        # Look for PRB event at the same slot as latest BSR
        events = self.event_processor.window_events[rnti]
        prb_time = None
        for i in range(bsr_indices[-1], -1, -1):
            event = events[i]
            if event[0] == EventTypes.PRB and event[4] == latest_bsr[4]:
                prb_time = event[3]
                break
        
        if prb_time is not None:
            # Check if there's an SR between last two BSRs
            has_sr = False
            earliest_sr_time = None
            prev_bsr_idx = bsr_indices[-2]
            
            for i in range(prev_bsr_idx, bsr_indices[-1]):
                if events[i][0] == EventTypes.SR:
                    has_sr = True
                    earliest_sr_time = events[i][3]
                    break
            
            # Adjust remaining time based on SR presence
            if has_sr and earliest_sr_time is not None:
                self.logger.log(f"remaining_time: {remaining_time}, latest_bsr[3]: {latest_bsr[3]}, earliest_sr_time: {earliest_sr_time}")
                remaining_time = (
                    remaining_time
                    - ((latest_bsr[3] - earliest_sr_time) * 1000 + 5)
                )
            else:
                self.logger.log(f"remaining_time: {remaining_time}, latest_bsr[3]: {latest_bsr[3]}, prb_time: {prb_time}")
                remaining_time = (
                    remaining_time
                    - ((latest_bsr[3] - prb_time) * 1000)
                )
        
        # Add the new request
        self.priority_manager.add_new_request(
            rnti, remaining_time, 
            self.event_processor.gnb_max_prb_slot, 
            int(latest_bsr[4])
        )
    
    def _log_prediction_result(
        self,
        rnti: int,
        final_prediction: bool,
        bsr_increased: bool,
        model_prediction: Optional[float]
    ) -> None:
        """Log the prediction result and current state.
        
        Args:
            rnti: Radio Network Temporary Identifier as integer.
            final_prediction: Final prediction result.
            bsr_increased: Whether BSR increased.
            model_prediction: Model prediction value (can be None).
        """
        timestamp = get_current_timestamp()
        
        log_msg = f"Prediction for RNTI 0x{rnti:x}: {final_prediction} at {timestamp}, "
        
        if model_prediction is not None:
            log_msg += f"Model_pred={model_prediction}, bsr_increased={bsr_increased}, "
        else:
            log_msg += f"bsr_increased={bsr_increased} (no model), "
        
        # Add remaining times summary
        summary = self.priority_manager.get_remaining_times_summary(rnti)
        log_msg += summary
        
        self.logger.log(log_msg)
    
    def get_ue_metrics_summary(self, rnti: int) -> Dict[str, Any]:
        """Get a summary of metrics for a specific UE.
        
        Args:
            rnti: Radio Network Temporary Identifier as integer.
            
        Returns:
            Dictionary with UE metrics summary.
        """
        summary = {
            "rnti": rnti,
            "bsr_events_count": 0,
            "peak_buffer_size": 0,
            "last_bsr": None,
            "current_priority": 0.0,
            "active_requests": 0
        }
        
        if rnti in self.ue_bsr_events:
            summary["bsr_events_count"] = len(self.ue_bsr_events[rnti])
        
        if rnti in self.ue_peak_buffer_size:
            summary["peak_buffer_size"] = self.ue_peak_buffer_size[rnti]
        
        if rnti in self.ue_last_bsr:
            summary["last_bsr"] = self.ue_last_bsr[rnti]
        
        if rnti in self.priority_manager.ue_priorities:
            summary["current_priority"] = self.priority_manager.ue_priorities[rnti]
        
        if rnti in self.priority_manager.ue_remaining_times:
            summary["active_requests"] = len(self.priority_manager.ue_remaining_times[rnti])
        
        return summary 