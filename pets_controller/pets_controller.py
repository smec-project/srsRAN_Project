import socket
import struct
import threading
import time
import math
from typing import Dict, Optional
import argparse
import numpy as np
from collections import deque


class PetsController:
    def __init__(
        self,
        app_port: int = 5557,  # Port to receive application messages
        ran_metrics_ip: str = "127.0.0.1",
        ran_metrics_port: int = 5556,  # Port to receive RAN metrics
        ran_control_ip: str = "127.0.0.1",
        ran_control_port: int = 5555,  # Port to send priority updates
        enable_logging: bool = False,  # Whether to enable logging
        window_size: int = 5,  # Size of the window for analysis
        model_path: str = None,  # Path to the trained model
        scaler_path: str = None,  # Path to the scaler
    ):
        # Application server setup
        self.app_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Enable address/port reuse
        self.app_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.app_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.app_socket.bind(("0.0.0.0", app_port))
        self.app_socket.listen(5)
        self.app_connections: Dict[str, socket.socket] = {}

        # RAN metrics connection setup
        self.ran_metrics_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ran_metrics_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.ran_metrics_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.ran_metrics_ip = ran_metrics_ip
        self.ran_metrics_port = ran_metrics_port

        # RAN control connection setup
        self.ran_control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ran_control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.ran_control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.ran_control_ip = ran_control_ip
        self.ran_control_port = ran_control_port

        # State tracking
        self.running = False
        self.current_metrics: Dict[str, dict] = {}  # RNTI (str) -> metrics
        self.ue_info: Dict[str, dict] = (
            {}
        )  # RNTI (str) -> {app_id, latency_req, request_size, ue_idx}
        self.request_sequences: Dict[str, list] = (
            {}
        )  # RNTI (str) -> list of sequence numbers

        # Change request_timers to store start timestamps
        self.request_start_times: Dict[str, Dict[int, float]] = (
            {}
        )  # RNTI -> {request_id -> start_timestamp}

        # Track UE priority states
        self.ue_priorities: Dict[str, float] = {}  # RNTI -> current_priority
        self.ue_bsr_events: Dict[str, deque] = (
            {}
        )  # RNTI -> deque of (bytes, slot) tuples
        self.ue_last_bsr: Dict[str, tuple] = {}  # RNTI -> (bytes, slot) of last BSR
        self.ue_peak_buffer_size: Dict[str, int] = {}  # RNTI -> peak buffer size

        # Track latest BSR state for each UE
        self.ue_latest_bsr: Dict[str, int] = {}  # RNTI -> latest BSR bytes

        # Track global gNB PRB allocation max slot
        self.gnb_max_prb_slot = 0

        # Add DDL tracking
        self.ue_ddl: Dict[str, float] = {}  # RNTI -> current DDL
        self.ue_last_priority: Dict[str, float] = (
            {}
        )  # RNTI -> last priority state (for reset tracking)
        self.MIN_DDL = 0.1  # Minimum DDL value (ms)

        # Window size for analysis
        self.window_size = window_size

        # Logging setup
        self.enable_logging = enable_logging
        self.log_file = open("controller.txt", "w") if enable_logging else None

        # Store events for each RNTI
        self.window_events: Dict[str, list] = {}  # RNTI -> list of events in window
        self.ue_slot_cycles_prb: Dict[str, int] = {}  # RNTI -> current PRB slot cycle
        self.ue_slot_cycles_ctrl: Dict[str, int] = (
            {}
        )  # RNTI -> current SR/BSR slot cycle
        self.ue_base_times: Dict[str, float] = {}  # RNTI -> base timestamp

        # Event type mapping
        self.EVENT_TYPES = {"SR": 0, "BSR": 1, "PRB": 2}

        # Load model and scaler if paths are provided
        self.model = None
        self.scaler = None
        if model_path and scaler_path:
            try:
                import joblib

                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                print(f"Loaded model from {model_path}")
                print(f"Loaded scaler from {scaler_path}")
            except Exception as e:
                print(f"Error loading model or scaler: {e}")

        # Counter for positive predictions
        self.positive_predictions: Dict[str, int] = (
            {}
        )  # RNTI -> count of positive predictions

        # Track request remaining times for each UE
        self.ue_remaining_times: Dict[str, list] = (
            {}
        )  # RNTI -> list of (request_id, remaining_time)
        self.ue_last_event_time: Dict[str, float] = (
            {}
        )  # RNTI -> timestamp of last event

        # Use global base slot for normalization
        self.global_base_slot = None  # Will be set by first event

    def start(self):
        """Start the controller and all its connections"""
        self.running = True

        # Connect to RAN services
        try:
            self.ran_metrics_socket.connect(
                (self.ran_metrics_ip, self.ran_metrics_port)
            )
            self.ran_control_socket.connect(
                (self.ran_control_ip, self.ran_control_port)
            )
        except Exception as e:
            self.log(f"Failed to connect to RAN services: {e}")
            return False

        # Start threads for different functionalities
        threading.Thread(target=self._handle_app_connections, daemon=True).start()
        threading.Thread(target=self._handle_ran_metrics, daemon=True).start()

        # Start priority update thread
        threading.Thread(target=self._update_priorities, daemon=True).start()
        return True

    def _update_priorities(self):
        """
        Update priorities based on the oldest request's remaining time
        - Priority increases as remaining time decreases
        - Reset priority when no requests pending
        - Skip priority updates for requests with latency requirements > 3s
        """
        while self.running:
            try:
                for rnti in list(self.ue_info.keys()):
                    if self.ue_priorities.get(rnti, -1) == -1:
                        self._initialize_ue_priority(rnti)

                    # Get the oldest request's remaining time
                    if (
                        rnti in self.ue_remaining_times
                        and self.ue_remaining_times[rnti]
                    ):
                        # Get first (oldest) request's remaining time
                        oldest_req = self.ue_remaining_times[rnti][0]
                        current_remaining = oldest_req[1]

                        # Convert remaining time from ms to s and calculate priority
                        remaining_seconds = current_remaining / 1000.0
                        current_bsr = self.ue_latest_bsr.get(rnti, 0)
                        priority = current_bsr / (
                            remaining_seconds * remaining_seconds + 1e-6
                        )
                        # priority = 1.0 / (remaining_seconds * remaining_seconds + 1e-6)

                        # Only update if priority changed
                        if self.ue_priorities.get(rnti, 0) != priority:
                            self.set_priority(rnti, priority)
                            self.ue_priorities[rnti] = priority
                            self.log(
                                f"Updated priority for RNTI {rnti}: Priority={priority:.2f}, "
                                f"Remaining_time={current_remaining:.2f}ms"
                            )
                    else:
                        # Reset priority if not already reset
                        if self.ue_priorities.get(rnti, 0) != 0:
                            self.reset_priority(rnti)
                            self.ue_priorities[rnti] = 0
                            self.log(f"Reset priority for RNTI {rnti}")

            except Exception as e:
                self.log(f"Error in priority update thread: {e}")

            # Sleep briefly to avoid busy waiting
            time.sleep(0.001)  # 1ms interval

    def _handle_app_connections(self):
        """Handle incoming application connections and messages"""
        while self.running:
            try:
                conn, addr = self.app_socket.accept()
                self.log(f"New application connected from {addr}")
                self.app_connections[addr[0]] = conn
                threading.Thread(
                    target=self._handle_app_messages, args=(conn, addr), daemon=True
                ).start()
            except Exception as e:
                self.log(f"Error accepting application connection: {e}")

    def _handle_app_messages(self, conn: socket.socket, addr):
        """Handle messages from a specific application connection"""
        try:
            # First message should be app registration with app_id
            data = conn.recv(1024)
            if not data:
                return

            app_id = data.decode("utf-8").strip()
            self.app_connections[app_id] = conn
            self.log(f"Application {app_id} registered from {addr}")

            while self.running:
                try:
                    data = conn.recv(1024)
                    if not data:
                        break

                    message = data.decode("utf-8").strip()
                    msg_parts = message.split("|")
                    msg_type = msg_parts[0]

                    if msg_type == "NEW_UE":
                        # Format: "NEW_UE|RNTI|UE_IDX|LATENCY_REQ|REQUEST_SIZE"
                        _, rnti, ue_idx, latency_req, request_size = msg_parts
                        # Store RNTI as string
                        self.ue_info[rnti] = {
                            "app_id": app_id,
                            "ue_idx": ue_idx,
                            "latency_req": float(latency_req),
                            "request_size": int(request_size),
                        }
                        self.request_sequences[rnti] = []
                        self.log(
                            f"New UE registered - RNTI: {rnti}, UE_IDX: {ue_idx}, "
                            f"Latency Req: {latency_req}ms, Size: {request_size} bytes"
                        )

                        self.request_start_times[rnti] = {}

                    elif msg_type == "Start":
                        # Format: "Start|rnti|seq_number"
                        _, rnti, seq_num = msg_parts
                        current_time = time.time()
                        self.log(
                            f"Request {seq_num} from RNTI {rnti} start at {current_time}"
                        )

                    elif msg_type == "REQUEST":
                        # Format: "REQUEST|RNTI|SEQ_NUM"
                        _, rnti, seq_num = msg_parts
                        seq_num = int(seq_num)

                        if rnti in self.ue_info:

                            # Store request start time
                            if rnti not in self.request_start_times:
                                self.request_start_times[rnti] = {}
                            self.request_start_times[rnti][seq_num] = time.time()
                        else:
                            self.log(f"Warning: Request for unknown RNTI {rnti}")

                    elif msg_type == "DONE":
                        # Format: "DONE|RNTI|SEQ_NUM"
                        _, rnti, seq_num = msg_parts
                        seq_num = int(seq_num)

                        # Calculate final processing time
                        if (
                            rnti in self.request_start_times
                            and seq_num in self.request_start_times[rnti]
                        ):
                            start_time = self.request_start_times[rnti][seq_num]
                            elapsed_time_ms = (
                                time.time() - start_time
                            ) * 1000  # Convert to ms
                            self.log(
                                f"Request {seq_num} from RNTI {rnti} completed in {elapsed_time_ms:.2f}ms at {time.time()}"
                            )
                            del self.request_start_times[rnti][seq_num]

                except Exception as e:
                    self.log(f"Error processing application message: {e}")
                    break

        finally:
            if app_id in self.app_connections:
                del self.app_connections[app_id]
            conn.close()

    def _handle_ran_metrics(self):
        """Receive and process RAN metrics"""
        while self.running:
            try:
                data = self.ran_metrics_socket.recv(1024).decode("utf-8")
                if not data:
                    continue

                for line in data.strip().split("\n"):
                    values = dict(item.split("=") for item in line.split(","))
                    msg_type = values["TYPE"]

                    if msg_type == "PRB":
                        self._handle_prb_metrics(values)
                    elif msg_type == "SR":
                        self._handle_sr_metrics(values)
                    elif msg_type == "BSR":
                        self._handle_bsr_metrics(values)

            except Exception as e:
                self.log(f"Error receiving RAN metrics: {e}")

    def set_priority(self, rnti: str, priority: float):
        """Send priority update to RAN"""
        try:
            # Format RNTI string and pack message in the correct format
            rnti_str = f"{rnti:<4}".encode("ascii")  # Left align, space pad to 4 chars
            msg = struct.pack("=5sdb", rnti_str, priority, False)

            self.ran_control_socket.send(msg)
            return True
        except Exception as e:
            self.log(f"Failed to send priority update: {e}")
            return False

    def reset_priority(self, rnti: str):
        """Reset priority for a specific RNTI"""
        try:
            # Reset priority state
            if rnti in self.ue_priorities:
                self.ue_priorities[rnti] = 0.0

            # Reset in scheduler
            rnti_str = f"{rnti:<4}".encode("ascii")
            msg = struct.pack("=5sdb", rnti_str, 0.0, True)
            self.ran_control_socket.send(msg)
            return True
        except Exception as e:
            self.log(f"Failed to reset priority: {e}")
            return False

    def stop(self):
        """Stop the controller and clean up connections"""
        self.running = False

        # Close all application connections
        for conn in self.app_connections.values():
            try:
                conn.shutdown(socket.SHUT_RDWR)
                conn.close()
            except:
                pass

        # Close server sockets
        try:
            self.app_socket.shutdown(socket.SHUT_RDWR)
            self.app_socket.close()
        except:
            pass

        try:
            self.ran_metrics_socket.shutdown(socket.SHUT_RDWR)
            self.ran_metrics_socket.close()
        except:
            pass

        try:
            self.ran_control_socket.shutdown(socket.SHUT_RDWR)
            self.ran_control_socket.close()
        except:
            pass

        # Close log file if it exists
        if self.log_file:
            self.log_file.close()

    def _initialize_ue_priority(self, rnti: str):
        """Initialize priority for a new UE"""
        self.ue_priorities[rnti] = 0.0

    def normalize_slot(self, rnti: str, slot: int, event_type: str) -> int:
        """
        Convert cyclic slots (0-20480) into a continuous increasing sequence
        relative to the global base slot
        """
        SLOT_MAX = 20480

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
                e for e in self.window_events[rnti] if e[0] == self.EVENT_TYPES["PRB"]
            ]
        else:  # SR or BSR
            cycles = self.ue_slot_cycles_ctrl
            # Get the previous control slot (SR or BSR)
            prev_events = [
                e
                for e in self.window_events[rnti]
                if e[0] in [self.EVENT_TYPES["SR"], self.EVENT_TYPES["BSR"]]
            ]

        if not prev_events:  # No previous events of this type
            return slot - self.global_base_slot

        # Get the previous raw slot
        prev_raw_slot = (
            prev_events[-1][4] + self.global_base_slot - cycles[rnti] * SLOT_MAX
        )

        # If new slot is less than previous slot, we've wrapped around
        if slot < prev_raw_slot:
            cycles[rnti] += 1

        absolute_slot = slot + (cycles[rnti] * SLOT_MAX)
        return absolute_slot - self.global_base_slot

    def add_event(
        self, rnti: str, event_type: str, timestamp: float, slot: int, **values
    ):
        """
        Add an event to the window and update if necessary
        Event format: [type, bytes, prbs, timestamp, slot, label]
        """
        if rnti not in self.window_events:
            self.window_events[rnti] = []
            self.ue_base_times[rnti] = timestamp
            self.ue_last_event_time[rnti] = timestamp

        # Update remaining times for all requests
        if rnti in self.ue_remaining_times and self.ue_remaining_times[rnti]:
            time_passed = (
                timestamp - self.ue_last_event_time[rnti]
            ) * 1000  # Convert to ms
            if time_passed > 0:
                # Update all remaining times
                updated_times = []
                for req_id, remaining in self.ue_remaining_times[rnti]:
                    new_remaining = remaining - time_passed
                    if new_remaining > 0:  # Only keep requests with remaining time > 0
                        updated_times.append((req_id, new_remaining))
                self.ue_remaining_times[rnti] = updated_times

        # Update last event time
        self.ue_last_event_time[rnti] = timestamp

        # Normalize slot
        normalized_slot = self.normalize_slot(rnti, slot, event_type)

        # Update global max PRB slot if this is a PRB event
        if event_type == "PRB":
            self.gnb_max_prb_slot = max(self.gnb_max_prb_slot, normalized_slot)

        # Create event array
        event = np.zeros(6, dtype=np.float32)
        event[0] = self.EVENT_TYPES[event_type]  # Event type
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
                and event[0] == self.EVENT_TYPES["PRB"]
                and e[0] == self.EVENT_TYPES["BSR"]
            ):
                insert_idx = i
                break
        self.window_events[rnti].insert(insert_idx, event)

        # Update window if it's a BSR event
        if event_type == "BSR":
            self.update_window(rnti)

    def print_window_data(self, rnti: str):
        """
        Print all events in the current window for a specific RNTI
        """
        if not self.window_events.get(rnti):
            return

        print(f"\nWindow data for RNTI {rnti}:")
        print("Type | Timestamp(ms) | BSR bytes | PRBs | Slot | Label")
        print("-" * 60)

        event_type_names = {v: k for k, v in self.EVENT_TYPES.items()}

        for event in self.window_events[rnti]:
            event_type = event_type_names[int(event[0])]
            timestamp_ms = event[3] * 1000  # Convert to milliseconds
            bsr_bytes = event[1]
            prbs = event[2]
            slot = event[4]
            label = event[5]

            print(
                f"{event_type:4} | {timestamp_ms:11.2f} | {bsr_bytes:9.0f} | {prbs:4.0f} | {slot:4.0f} | {label:5.0f}"
            )

        print("-" * 60)

    def extract_window_features(self, events, window_bsr_indices):
        """
        Extract features for a window between BSRs
        Returns features in the same order as training
        """
        all_features = []

        # Get the last BSR's slot as the search end slot
        final_bsr_slot = events[window_bsr_indices[-1]][4]

        # Process each BSR interval in the window
        for i in range(len(window_bsr_indices) - 1):
            current_start_idx = window_bsr_indices[i]
            current_end_idx = window_bsr_indices[i + 1]

            start_bsr = events[current_start_idx]
            end_bsr = events[current_end_idx]

            # Find first PRB after start_bsr but before final_bsr
            first_prb_slot = final_bsr_slot  # Default to final BSR slot if no PRB found
            search_idx = current_start_idx + 1
            while search_idx < len(events):
                if events[search_idx][0] == self.EVENT_TYPES["PRB"]:
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

            for event in events[current_start_idx + 1 : current_end_idx]:
                if event[0] == self.EVENT_TYPES["PRB"]:
                    total_prbs += event[2]
                    prb_events += 1
                elif event[0] == self.EVENT_TYPES["SR"]:
                    sr_count += 1

            # Calculate window duration in slots
            window_slots = end_bsr[4] - start_bsr[4]

            # Calculate BSR per PRB
            bsr_per_prb = bsr_diff / (total_prbs + 1e-6)

            # Calculate rates using slots (multiply by 1000 to convert to per-millisecond rate)
            window_duration_ms = window_slots * 0.5  # Assuming each slot is 0.5ms
            bsr_update_rate = 1000.0 / (window_duration_ms + 1e-6)
            sr_rate = sr_count * 1000.0 / (window_duration_ms + 1e-6)

            # Features in the same order as training
            interval_features = np.array(
                [
                    bsr_diff,  # Difference between BSRs
                    total_prbs,  # Total PRBs allocated
                    sr_count,  # Number of SR events
                    end_bsr_value,  # Value of the end BSR
                    bsr_per_prb,  # BSR difference normalized by PRBs
                    window_slots,  # Time duration in slots
                    bsr_update_rate,  # Rate of BSR updates
                    sr_rate,  # Rate of SR events
                    slots_until_prb,  # Slots until first PRB after BSR
                ]
            )

            all_features.append(interval_features)

        return np.concatenate(all_features)

    def update_window(self, rnti: str):
        """Update the sliding window when a new BSR event arrives"""
        # Find all BSR indices in current events
        bsr_indices = [
            i
            for i, e in enumerate(self.window_events[rnti])
            if e[0] == self.EVENT_TYPES["BSR"]
        ]

        if len(bsr_indices) > self.window_size:
            # Window full, remove events before the window
            start_idx = bsr_indices[-(self.window_size + 1)]
            self.window_events[rnti] = self.window_events[rnti][start_idx:]

            # If model is loaded, do inference
            if self.model is not None and self.scaler is not None:
                # Get new BSR indices after window update
                new_bsr_indices = [
                    i
                    for i, e in enumerate(self.window_events[rnti])
                    if e[0] == self.EVENT_TYPES["BSR"]
                ]

                # Check if latest BSR bytes increased
                latest_bsr = self.window_events[rnti][new_bsr_indices[-1]]
                prev_bsr = self.window_events[rnti][new_bsr_indices[-2]]
                bsr_increased = latest_bsr[1] > prev_bsr[1]

                # Get model prediction
                features = self.extract_window_features(
                    self.window_events[rnti], new_bsr_indices
                )
                features_scaled = self.scaler.transform(features.reshape(1, -1))
                model_prediction = self.model.predict(features_scaled)[0]

                # Final prediction is OR of model prediction and BSR increase
                prediction = int(bool(model_prediction) or bool(bsr_increased))

                # Get latest BSR value
                latest_bsr = self.window_events[rnti][new_bsr_indices[-1]]

                # If BSR is 0, clear all remaining times
                if latest_bsr[1] == 0:
                    if rnti in self.ue_remaining_times:
                        self.ue_remaining_times[rnti] = []
                else:
                    # If prediction is 1, add new request with initial remaining time
                    if prediction:
                        # Update positive prediction counter
                        if rnti not in self.positive_predictions:
                            self.positive_predictions[rnti] = 0
                        self.positive_predictions[rnti] += 1

                        # Calculate initial remaining time
                        remaining_time = self.ue_info[rnti]["latency_req"]
                        prb_time = None
                        for i in range(new_bsr_indices[-1], -1, -1):
                            event = self.window_events[rnti][i]
                            if (
                                event[0] == self.EVENT_TYPES["PRB"]
                                and event[4] == latest_bsr[4]
                            ):
                                prb_time = event[3]
                                break

                        if prb_time is not None:
                            # Check if there's an SR between last two BSRs and get earliest SR time
                            has_sr = False
                            earliest_sr_time = None
                            prev_bsr_idx = new_bsr_indices[-2]
                            for i in range(prev_bsr_idx, new_bsr_indices[-1]):
                                if (
                                    self.window_events[rnti][i][0]
                                    == self.EVENT_TYPES["SR"]
                                ):
                                    has_sr = True
                                    earliest_sr_time = self.window_events[rnti][i][3]
                                    break

                            if has_sr:
                                remaining_time = (
                                    remaining_time
                                    - ((latest_bsr[3] - earliest_sr_time) * 1000 + 5)
                                    - (self.gnb_max_prb_slot - latest_bsr[4]) * 0.5
                                )
                            else:
                                remaining_time = (
                                    remaining_time
                                    - ((latest_bsr[3] - prb_time) * 1000)
                                    - (self.gnb_max_prb_slot - latest_bsr[4]) * 0.5
                                )

                        # Add new request with its remaining time, using timestamp as ID
                        if rnti not in self.ue_remaining_times:
                            self.ue_remaining_times[rnti] = []
                        self.ue_remaining_times[rnti].append(
                            (self.positive_predictions[rnti], remaining_time)
                        )  # Use timestamp as request ID

                    # Log prediction and all remaining times
                    log_msg = (
                        f"Prediction for RNTI {rnti}: {prediction} at {time.time()}, "
                    )
                    log_msg += f"Model_pred={model_prediction}, bsr_increased={bsr_increased}, "
                    log_msg += f"Total positive predictions: {self.positive_predictions.get(rnti, 0)}"
                    if rnti in self.ue_remaining_times:
                        for req_label, remaining in self.ue_remaining_times[rnti]:
                            log_msg += (
                                f", Request_{req_label}_remaining={remaining:.2f}ms"
                            )
                    self.log(log_msg)

    def _handle_sr_metrics(self, values):
        """Handle SR indication metrics"""
        rnti = values["RNTI"][-4:]
        slot = int(values["SLOT"])
        self.log(f"SR received from RNTI=0x{rnti}, slot={slot} at {time.time()}")
        self.add_event(rnti, "SR", time.time(), slot)

    def _handle_bsr_metrics(self, values):
        """Handle BSR metrics"""
        rnti = values["RNTI"][-4:]
        slot = int(values["SLOT"])
        bytes_val = int(values["BYTES"])

        # Update latest BSR state
        self.ue_latest_bsr[rnti] = bytes_val

        # Initialize FIFO queue if not exists
        if rnti not in self.ue_bsr_events:
            self.ue_bsr_events[rnti] = deque()

        # Add new BSR event to queue
        self.ue_bsr_events[rnti].append((bytes_val, slot))

        self.log(
            f"bsr received from RNTI=0x{rnti}, slot={slot}, bytes={bytes_val} at {time.time()}"
        )
        self.add_event(rnti, "BSR", time.time(), slot, bytes=bytes_val)

    def _handle_prb_metrics(self, values):
        """Handle PRB allocation metrics"""
        rnti = values["RNTI"][-4:]
        slot = int(values["SLOT"])
        prbs = int(values["PRBs"])
        self.log(
            f"PRB received from RNTI=0x{rnti}, slot={slot}, prbs={prbs} at {time.time()}"
        )
        self.add_event(rnti, "PRB", time.time(), slot, prbs=prbs)

    def log(self, message: str):
        """Helper method for logging"""
        if self.enable_logging and self.log_file:
            self.log_file.write(message + "\n")
            self.log_file.flush()


def main():
    parser = argparse.ArgumentParser(description="PETS Controller")
    parser.add_argument(
        "--log",
        action="store_true",
        default=False,
        help="Enable logging to file (default: False)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Size of the window for analysis (default: 5)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="labeled_data/models/bsr_only_xgboost.joblib",
        help="Path to the trained model for inference (default: labeled_data/bsr_only_xgboost.joblib)",
    )
    parser.add_argument(
        "--scaler-path",
        type=str,
        default="labeled_data/models/bsr_only_scaler.joblib",
        help="Path to the scaler for the model (default: labeled_data/bsr_only_scaler.joblib)",
    )
    args = parser.parse_args()

    controller = PetsController(
        enable_logging=args.log,
        window_size=args.window_size,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
    )

    if controller.start():
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down controller...")
        finally:
            controller.stop()


if __name__ == "__main__":
    main()
