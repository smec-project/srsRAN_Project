"""Event processing and windowing functionality for PMEC Controller."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque

from .config import EventTypes, NetworkConstants
from .utils import Logger, get_current_timestamp


class EventProcessor:
    """Handles event processing, windowing, and slot normalization.
    
    Manages sliding windows of events for each UE and provides
    feature extraction capabilities for machine learning inference.
    """
    
    def __init__(self, window_size: int, logger: Logger):
        """Initialize the event processor.
        
        Args:
            window_size: Size of the sliding window for analysis.
            logger: Logger instance for debugging output.
        """
        self.window_size = window_size
        self.logger = logger
        
        # Store events for each RNTI
        self.window_events: Dict[str, List[np.ndarray]] = {}
        
        # Slot cycle tracking for each UE and event type
        self.ue_slot_cycles_prb: Dict[str, int] = {}
        self.ue_slot_cycles_ctrl: Dict[str, int] = {}
        
        # Base times for relative timestamp calculation
        self.ue_base_times: Dict[str, float] = {}
        
        # Global base slot for normalization
        self.global_base_slot: Optional[int] = None
        
        # Track maximum PRB slot globally
        self.gnb_max_prb_slot: int = 0
    
    def normalize_slot(self, rnti: str, slot: int, event_type: str) -> int:
        """Convert cyclic slots (0-20480) into a continuous increasing sequence.
        
        Args:
            rnti: Radio Network Temporary Identifier.
            slot: Current slot number (0-20480).
            event_type: Type of event ("PRB", "SR", "BSR").
            
        Returns:
            Normalized slot number relative to global base slot.
        """
        if self.global_base_slot is None:
            self.global_base_slot = slot
            return 0

        if rnti not in self.ue_slot_cycles_prb:
            self.ue_slot_cycles_prb[rnti] = 0
            self.ue_slot_cycles_ctrl[rnti] = 0
            return 0

        # Choose the appropriate cycle counter based on event type
        if event_type == "PRB":
            cycles = self.ue_slot_cycles_prb
            # Get the previous PRB slot
            prev_events = [
                e for e in self.window_events[rnti]
                if e[0] == EventTypes.PRB
            ]
        else:  # SR or BSR
            cycles = self.ue_slot_cycles_ctrl
            # Get the previous control slot (SR or BSR)
            prev_events = [
                e for e in self.window_events[rnti]
                if e[0] in [EventTypes.SR, EventTypes.BSR]
            ]

        if not prev_events:  # No previous events of this type
            return slot - self.global_base_slot

        # Get the previous raw slot
        prev_raw_slot = (
            prev_events[-1][4] + self.global_base_slot - 
            cycles[rnti] * NetworkConstants.SLOT_MAX
        )

        # If new slot is less than previous slot, we've wrapped around
        if slot < prev_raw_slot:
            cycles[rnti] += 1

        absolute_slot = slot + (cycles[rnti] * NetworkConstants.SLOT_MAX)
        return absolute_slot - self.global_base_slot
    
    def add_event(
        self, 
        rnti: str, 
        event_type: str, 
        timestamp: float, 
        slot: int, 
        **values
    ) -> None:
        """Add an event to the window for a specific UE.
        
        Args:
            rnti: Radio Network Temporary Identifier.
            event_type: Type of event ("PRB", "SR", "BSR").
            timestamp: Timestamp when the event occurred.
            slot: Slot number when the event occurred.
            **values: Additional event-specific values (bytes, prbs, etc.).
        """
        if rnti not in self.window_events:
            self.window_events[rnti] = []
            self.ue_base_times[rnti] = timestamp

        # Normalize slot
        normalized_slot = self.normalize_slot(rnti, slot, event_type)

        # Update global max PRB slot if this is a PRB event
        if event_type == "PRB":
            self.gnb_max_prb_slot = max(self.gnb_max_prb_slot, normalized_slot)

        # Create event array: [type, bytes, prbs, timestamp, slot, label]
        event = np.zeros(6, dtype=np.float32)
        event[0] = getattr(EventTypes, event_type)  # Event type
        event[1] = values.get("bytes", 0)  # BSR bytes
        event[2] = values.get("prbs", 0)  # PRBs
        event[3] = timestamp - self.ue_base_times[rnti]  # Relative timestamp
        event[4] = normalized_slot  # Normalized slot
        event[5] = 0  # Label (not used in controller)

        # Insert event in sorted position based on slot
        insert_idx = len(self.window_events[rnti])
        for i, e in enumerate(self.window_events[rnti]):
            if e[4] > normalized_slot or (
                e[4] == normalized_slot
                and event[0] == EventTypes.PRB
                and e[0] == EventTypes.BSR
            ):
                insert_idx = i
                break
        self.window_events[rnti].insert(insert_idx, event)
    
    def get_bsr_indices(self, rnti: str) -> List[int]:
        """Get indices of all BSR events for a specific UE.
        
        Args:
            rnti: Radio Network Temporary Identifier.
            
        Returns:
            List of indices where BSR events occur.
        """
        if rnti not in self.window_events:
            return []
        
        return [
            i for i, e in enumerate(self.window_events[rnti])
            if e[0] == EventTypes.BSR
        ]
    
    def trim_window(self, rnti: str) -> bool:
        """Trim the sliding window to maintain the specified size.
        
        Args:
            rnti: Radio Network Temporary Identifier.
            
        Returns:
            True if window was trimmed and is ready for analysis.
        """
        bsr_indices = self.get_bsr_indices(rnti)
        
        if len(bsr_indices) > self.window_size:
            # Window full, remove events before the window
            start_idx = bsr_indices[-(self.window_size + 1)]
            self.window_events[rnti] = self.window_events[rnti][start_idx:]
            return True
        
        return False
    
    def extract_window_features(self, rnti: str) -> Optional[np.ndarray]:
        """Extract features for machine learning inference.
        
        Args:
            rnti: Radio Network Temporary Identifier.
            
        Returns:
            Feature array for ML model or None if insufficient data.
        """
        if rnti not in self.window_events:
            return None
        
        events = self.window_events[rnti]
        bsr_indices = self.get_bsr_indices(rnti)
        
        if len(bsr_indices) < 2:
            return None
        
        all_features = []
        
        # Get the last BSR's slot as the search end slot
        final_bsr_slot = events[bsr_indices[-1]][4]

        # Process each BSR interval in the window
        for i in range(len(bsr_indices) - 1):
            current_start_idx = bsr_indices[i]
            current_end_idx = bsr_indices[i + 1]

            start_bsr = events[current_start_idx]
            end_bsr = events[current_end_idx]

            # Find first PRB after start_bsr but before final_bsr
            first_prb_slot = final_bsr_slot  # Default to final BSR slot if no PRB found
            search_idx = current_start_idx + 1
            while search_idx < len(events):
                if events[search_idx][0] == EventTypes.PRB:
                    first_prb_slot = events[search_idx][4]
                    break
                if events[search_idx][4] > final_bsr_slot:
                    break
                search_idx += 1

            # Calculate slot difference until first PRB
            slots_until_prb = first_prb_slot - start_bsr[4]

            # Calculate other features
            bsr_diff = end_bsr[1] - start_bsr[1]
            end_bsr_value = end_bsr[1]

            total_prbs = 0
            sr_count = 0
            prb_events = 0

            for event in events[current_start_idx + 1:current_end_idx]:
                if event[0] == EventTypes.PRB:
                    total_prbs += event[2]
                    prb_events += 1
                elif event[0] == EventTypes.SR:
                    sr_count += 1

            # Calculate window duration in slots
            window_slots = end_bsr[4] - start_bsr[4]

            # Calculate BSR per PRB (avoid division by zero)
            bsr_per_prb = bsr_diff / (total_prbs + 1e-6)

            # Calculate rates using slots
            window_duration_ms = window_slots * 0.5  # Assuming each slot is 0.5ms
            bsr_update_rate = 1000.0 / (window_duration_ms + 1e-6)
            sr_rate = sr_count * 1000.0 / (window_duration_ms + 1e-6)

            # Features in the same order as training
            interval_features = np.array([
                bsr_diff,  # Difference between BSRs
                total_prbs,  # Total PRBs allocated
                sr_count,  # Number of SR events
                end_bsr_value,  # Value of the end BSR
                bsr_per_prb,  # BSR difference normalized by PRBs
                window_slots,  # Time duration in slots
                bsr_update_rate,  # Rate of BSR updates
                sr_rate,  # Rate of SR events
                slots_until_prb,  # Slots until first PRB after BSR
            ])

            all_features.append(interval_features)

        return np.concatenate(all_features)
    
    def print_window_data(self, rnti: str) -> None:
        """Print all events in the current window for debugging.
        
        Args:
            rnti: Radio Network Temporary Identifier.
        """
        if not self.window_events.get(rnti):
            return

        print(f"\nWindow data for RNTI {rnti}:")
        print("Type | Timestamp(ms) | BSR bytes | PRBs | Slot | Label")
        print("-" * 60)

        event_type_names = EventTypes.get_name_mapping()
        event_type_names = {v: k for k, v in event_type_names.items()}

        for event in self.window_events[rnti]:
            event_type = event_type_names[int(event[0])]
            timestamp_ms = event[3] * 1000  # Convert to milliseconds
            bsr_bytes = event[1]
            prbs = event[2]
            slot = event[4]
            label = event[5]

            print(
                f"{event_type:4} | {timestamp_ms:11.2f} | {bsr_bytes:9.0f} |"
                f" {prbs:4.0f} | {slot:4.0f} | {label:5.0f}"
            )

        print("-" * 60) 