"""Event processing and windowing functionality for SMEC Controller."""

import numpy as np
from typing import Dict, List, Optional

from .config import EventTypes, NetworkConstants
from .utils import Logger


class EventProcessor:
    """Handles event processing, windowing, and slot normalization.

    Manages sliding windows of events for each UE.
    """
    
    def __init__(self, window_size: int, logger: Logger):
        """Initialize the event processor.
        
        Args:
            window_size: Size of the sliding window for analysis.
            logger: Logger instance for debugging output.
        """
        self.window_size = window_size
        self.logger = logger
        
        # Store events for each RNTI - using int RNTI as keys
        self.window_events: Dict[int, List[np.ndarray]] = {}
        
        # Slot cycle tracking for each UE and event type
        self.ue_slot_cycles_prb: Dict[int, int] = {}
        self.ue_slot_cycles_ctrl: Dict[int, int] = {}
        
        # Base times for relative timestamp calculation
        self.ue_base_times: Dict[int, float] = {}
        
        # Global base slot for normalization
        self.global_base_slot: Optional[int] = None
        
        # Track maximum PRB slot globally
        self.gnb_max_prb_slot: int = 0
    
    def normalize_slot(self, rnti: int, slot: int, event_type: str) -> int:
        """Convert cyclic slots (0-20480) into a continuous increasing sequence.
        
        Args:
            rnti: Radio Network Temporary Identifier as integer.
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
        if slot < prev_raw_slot and prev_raw_slot - slot > 10240:
            cycles[rnti] += 1

        absolute_slot = slot + (cycles[rnti] * NetworkConstants.SLOT_MAX)
        return absolute_slot - self.global_base_slot
    
    def add_event(
        self, 
        rnti: int, 
        event_type: str, 
        timestamp: float, 
        slot: int, 
        **values
    ) -> None:
        """Add an event to the window for a specific UE.
        
        Args:
            rnti: Radio Network Temporary Identifier as integer.
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
    
    def get_bsr_indices(self, rnti: int) -> List[int]:
        """Get indices of all BSR events for a specific UE.
        
        Args:
            rnti: Radio Network Temporary Identifier as integer.
            
        Returns:
            List of indices where BSR events occur.
        """
        if rnti not in self.window_events:
            return []
        
        return [
            i for i, e in enumerate(self.window_events[rnti])
            if e[0] == EventTypes.BSR
        ]
    
    def trim_window(self, rnti: int) -> bool:
        """Trim the sliding window to maintain the specified size.
        
        Args:
            rnti: Radio Network Temporary Identifier as integer.
            
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

    def print_window_data(self, rnti: int) -> None:
        """Print all events in the current window for debugging.
        
        Args:
            rnti: Radio Network Temporary Identifier as integer.
        """
        if not self.window_events.get(rnti):
            return

        print(f"\nWindow data for RNTI 0x{rnti:x}:")
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