import socket
import struct
import threading
import time
import math
from typing import Dict, Optional

# Import message types from main controller
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from config import MessageTypes


class TrainDataCollector:
    def __init__(
        self,
        slo_ctrl_port: int = 5557,  # Port to receive SLO control plane messages
        ran_metrics_ip: str = "127.0.0.1",
        ran_metrics_port: int = 5556,  # Port to receive RAN metrics
        ran_control_ip: str = "127.0.0.1",
        ran_control_port: int = 5555,  # Port to send priority updates
    ):
        # SLO control plane server setup
        self.slo_ctrl_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Enable address/port reuse
        self.slo_ctrl_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.slo_ctrl_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.slo_ctrl_socket.bind(("0.0.0.0", slo_ctrl_port))
        self.slo_ctrl_socket.listen(5)
        self.slo_ctrl_connections: Dict[str, socket.socket] = {}

        # RAN metrics connection setup
        self.ran_metrics_socket = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        )
        self.ran_metrics_socket.setsockopt(
            socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
        )
        self.ran_metrics_socket.setsockopt(
            socket.SOL_SOCKET, socket.SO_REUSEPORT, 1
        )
        self.ran_metrics_ip = ran_metrics_ip
        self.ran_metrics_port = ran_metrics_port

        # RAN control connection setup
        self.ran_control_socket = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        )
        self.ran_control_socket.setsockopt(
            socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
        )
        self.ran_control_socket.setsockopt(
            socket.SOL_SOCKET, socket.SO_REUSEPORT, 1
        )
        self.ran_control_ip = ran_control_ip
        self.ran_control_port = ran_control_port

        # State tracking
        self.running = False
        self.current_metrics: Dict[int, dict] = {}  # RNTI (int) -> metrics
        self.ue_info: Dict[int, dict] = (
            {}
        )  # RNTI (int) -> {slo_latency}
        self.request_sequences: Dict[int, list] = (
            {}
        )  # RNTI (int) -> list of sequence numbers

        # Track resource requirements and pending requests for each UE
        self.ue_resource_needs: Dict[int, int] = (
            {}
        )  # RNTI -> total bytes needed
        self.ue_pending_requests: Dict[int, Dict[int, int]] = (
            {}
        )  # RNTI -> {request_id -> bytes}

        # Track latest PRB allocation and slot for each UE
        self.ue_prb_status: Dict[int, tuple] = {}  # RNTI -> (slot, prbs)

        # Change request_timers to store start timestamps
        self.request_start_times: Dict[int, Dict[int, float]] = (
            {}
        )  # RNTI -> {request_id -> start_timestamp}

        # Track UE priority states
        self.ue_priorities: Dict[int, float] = {}  # RNTI -> current_priority

        # Track PRB allocation history
        self.HISTORY_WINDOW = 100  # Keep last 100 slot records
        self.prb_history: Dict[int, Dict[int, int]] = (
            {}
        )  # RNTI -> {slot -> prbs}
        self.slot_history: list = []  # Ordered list of slots we've seen

        # Track allocated PRBs for each request
        self.request_prb_allocations: Dict[int, Dict[int, int]] = (
            {}
        )  # RNTI -> {request_id -> total_allocated_prbs}

        # Open log file first
        self.log_file = open("controller.txt", "w")

    def start(self):
        """
        Start the controller and all its connections.
        """
        self.running = True

        # Connect to RAN services
        try:
            self.ran_metrics_socket.connect(
                (self.ran_metrics_ip, self.ran_metrics_port)
            )
            # self.ran_control_socket.connect((self.ran_control_ip, self.ran_control_port))
        except Exception as e:
            self.log_file.write(f"Failed to connect to RAN services: {e}\n")
            self.log_file.flush()
            return False

        # Start threads for different functionalities
        threading.Thread(
            target=self._handle_slo_ctrl_connections, daemon=True
        ).start()
        threading.Thread(target=self._handle_ran_metrics, daemon=True).start()

        return True

    def _handle_slo_ctrl_connections(self):
        """
        Handle incoming SLO control plane connections and messages.
        """
        while self.running:
            try:
                conn, addr = self.slo_ctrl_socket.accept()
                self.log_file.write(f"New SLO control plane connected from {addr}\n")
                self.log_file.flush()
                self.slo_ctrl_connections[addr[0]] = conn
                threading.Thread(
                    target=self._handle_slo_ctrl_messages,
                    args=(conn, addr),
                    daemon=True,
                ).start()
            except Exception as e:
                self.log_file.write(
                    f"Error accepting SLO control plane connection: {e}\n"
                )
                self.log_file.flush()

    def _handle_slo_ctrl_messages(self, conn: socket.socket, addr):
        """
        Handle binary messages from SLO control plane.
        """
        try:
            self.log_file.write(f"SLO control plane client connected from {addr}\n")
            
            while self.running:
                try:
                    data = conn.recv(1024)
                    if not data:
                        break

                    self._process_slo_ctrl_message(data)

                except Exception as e:
                    self.log_file.write(f"Error processing SLO control plane message: {e}\n")
                    self.log_file.flush()
                    break

        finally:
            conn.close()

    def _process_slo_ctrl_message(self, data: bytes) -> None:
        """Process a binary message from SLO control plane.
        
        Args:
            data: The binary message data to process.
        """
        try:
            # Expect 12 bytes: uint32 (message_type) + uint32 (rnti) + uint32 (slo_latency)
            if len(data) != 12:
                self.log_file.write(f"Invalid SLO control message size: {len(data)} bytes, expected 12\n")
                self.log_file.flush()
                return
            
            # Unpack: uint32 message_type, uint32 RNTI, uint32 SLO latency
            msg_type, rnti, slo_latency_uint = struct.unpack('=III', data)
            
            if msg_type == MessageTypes.SLO_MESSAGE:
                # Convert uint32 SLO latency to float
                slo_latency_float = float(slo_latency_uint)
                
                # Store UE info with RNTI as integer
                self.ue_info[rnti] = {
                    "slo_latency": slo_latency_float,
                }
                self.request_sequences[rnti] = []
                
                self.log_file.write(
                    f"New UE registered - RNTI: 0x{rnti:x}, SLO Latency: {slo_latency_float}ms\n"
                )
                self.log_file.flush()
                
                # Initialize resource tracking for new UE
                self.ue_resource_needs[rnti] = 0
                self.ue_pending_requests[rnti] = {}
                self.request_start_times[rnti] = {}
                
            else:
                self.log_file.write(f"Unknown message type: {msg_type}\n")
                self.log_file.flush()
            
        except Exception as e:
            self.log_file.write(f"Error processing SLO control plane binary message: {e}\n")
            self.log_file.flush()

    def _handle_ran_metrics(self):
        """
        Receive and process RAN metrics.
        """
        while self.running:
            try:
                data = self.ran_metrics_socket.recv(1024)
                if not data:
                    continue

                self._process_ran_metrics_message(data)

            except Exception as e:
                self.log_file.write(f"Error receiving RAN metrics: {e}\n")
                self.log_file.flush()

    def _process_ran_metrics_message(self, data: bytes) -> None:
        """Process binary RAN metrics message.
        
        Args:
            data: The binary message data to process.
        """
        try:
            # Expect 16 bytes: uint32 (type) + uint32 (rnti) + uint32 (slot) + uint32 (value)
            if len(data) != 16:
                self.log_file.write(f"Invalid RAN metrics message size: {len(data)} bytes, expected 16\n")
                self.log_file.flush()
                return
            
            # Unpack: uint32 type, uint32 RNTI, uint32 slot, uint32 value
            msg_type, rnti, slot, value = struct.unpack('=IIII', data)
            
            if msg_type == 0:  # PRB message
                self._handle_prb_metrics(rnti, slot, value)
            elif msg_type == 1:  # SR message
                self._handle_sr_metrics(rnti, slot)
            elif msg_type == 2:  # BSR message
                self._handle_bsr_metrics(rnti, slot, value)
            else:
                self.log_file.write(f"Unknown RAN metrics type: {msg_type}\n")
                self.log_file.flush()
                
        except Exception as e:
            self.log_file.write(f"Error processing RAN metrics binary message: {e}\n")
            self.log_file.flush()

    def set_priority(self, rnti: int, priority: float):
        """
        Send priority update to RAN.
        """
        try:
            # Pack message with uint32 RNTI, float priority, bool reset
            msg = struct.pack("=Ifb", rnti, priority, False)

            self.ran_control_socket.send(msg)
            return True
        except Exception as e:
            self.log_file.write(f"Failed to send priority update: {e}\n")
            self.log_file.flush()
            return False

    def reset_priority(self, rnti: int):
        """
        Reset priority for a specific RNTI.
        """
        try:
            # Reset priority state
            if rnti in self.ue_priorities:
                self.ue_priorities[rnti] = 0.0

            # Reset in scheduler
            msg = struct.pack("=Ifb", rnti, 0.0, True)
            self.ran_control_socket.send(msg)
            return True
        except Exception as e:
            self.log_file.write(f"Failed to reset priority: {e}\n")
            self.log_file.flush()
            return False

    def stop(self):
        """
        Stop the controller and clean up connections.
        """
        self.running = False

        # Close all SLO control plane connections
        for conn in self.slo_ctrl_connections.values():
            try:
                conn.shutdown(socket.SHUT_RDWR)
                conn.close()
            except:
                pass

        # Close server sockets
        try:
            self.slo_ctrl_socket.shutdown(socket.SHUT_RDWR)
            self.slo_ctrl_socket.close()
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

    def _initialize_ue_priority(self, rnti: int):
        """
        Initialize priority for a new UE.
        """
        self.ue_priorities[rnti] = 0.0

    def _calculate_incentive_priority(self, ue_rnti: int) -> float:
        """
        Calculate priority for incentive mode (first half of latency
        requirement)
        """
        # Constants
        BYTES_PER_PRB = 90
        TTI_DURATION = 2.5  # ms
        DEFAULT_PRIORITY_OFFSET = 1.0  # Initial priority offset

        # Store calculations for each UE with requests
        ue_prb_requirements = {}  # rnti -> (total_prbs, prbs_per_tti)

        # Calculate for each UE with active requests
        for rnti in self.request_start_times.keys():
            if not self.request_start_times[rnti]:
                continue

            earliest_req_id = min(
                self.request_start_times[rnti].items(), key=lambda x: x[1]
            )[0]
            request_size = self.ue_pending_requests[rnti][earliest_req_id]
            total_prbs = (request_size + BYTES_PER_PRB - 1) // BYTES_PER_PRB
            slo_latency = self.ue_info[rnti]["slo_latency"]
            available_ttis = slo_latency / TTI_DURATION
            prbs_per_tti = (total_prbs + int(available_ttis) - 1) // int(
                available_ttis
            )
            ue_prb_requirements[rnti] = (total_prbs, prbs_per_tti)

        # Get the latest slot from history
        if not self.slot_history:
            return DEFAULT_PRIORITY_OFFSET
        latest_slot = self.slot_history[-1]

        # Calculate priority adjustments based on PRB differences
        priority_adjustments = {}
        for rnti, (_, required_prbs_per_tti) in ue_prb_requirements.items():
            if rnti not in self.prb_history:
                self.log_file.write(
                    f"Warning: No PRB history for RNTI {rnti}\n"
                )
                self.log_file.flush()
                return DEFAULT_PRIORITY_OFFSET
            actual_prbs = self.prb_history[rnti][latest_slot]
            prb_difference = actual_prbs - required_prbs_per_tti
            current_offset = self.ue_priorities.get(
                rnti, DEFAULT_PRIORITY_OFFSET
            )
            priority_adjustments[rnti] = prb_difference * current_offset
        total_priority_metric = sum(priority_adjustments.values())
        ue_priority_metric = priority_adjustments[ue_rnti]
        current_offset = self.ue_priorities[ue_rnti]
        if ue_priority_metric > 0 and total_priority_metric < 0:
            current_offset = current_offset / 2
        else:
            actual_prbs = self.prb_history[ue_rnti][latest_slot]
            if actual_prbs > 0:
                current_offset = self.ue_priorities[ue_rnti] + max(
                    DEFAULT_PRIORITY_OFFSET,
                    abs(actual_prbs - ue_prb_requirements[ue_rnti][1])
                    / actual_prbs,
                )
            else:
                current_offset = self.ue_priorities[ue_rnti] + abs(
                    actual_prbs - ue_prb_requirements[ue_rnti][1]
                )
        # self.log_file.write(f"incentive {ue_rnti} {actual_prbs} {ue_prb_requirements[ue_rnti][1]} {current_offset}\n")
        # self.log_file.flush()  # Ensure immediate write
        return current_offset

    def _calculate_accelerate_priority(self, ue_rnti: int) -> float:
        """
        Calculate priority for accelerate mode (second half of latency
        requirement)
        """
        # Constants
        BYTES_PER_PRB = 90
        MS_TO_S = 0.001  # Convert ms to s

        # Get current request info
        if not self.request_start_times[ue_rnti]:
            return 0.0

        # Get earliest request and its timer
        earliest_req_id = min(
            self.request_start_times[ue_rnti].items(), key=lambda x: x[1]
        )[0]
        start_time = self.request_start_times[ue_rnti][earliest_req_id]
        elapsed_time_ms = (time.time() - start_time) * 1000
        slo_latency_ms = self.ue_info[ue_rnti]["slo_latency"]

        # Calculate time to deadline in seconds
        time_to_deadline_s = (slo_latency_ms - elapsed_time_ms) * MS_TO_S

        # Calculate remaining PRBs needed
        total_prbs_needed = (
            self.ue_pending_requests[ue_rnti][earliest_req_id]
            + BYTES_PER_PRB
            - 1
        ) // BYTES_PER_PRB

        # Safely get allocated PRBs
        if ue_rnti not in self.request_prb_allocations:
            self.request_prb_allocations[ue_rnti] = {}
        prbs_allocated = self.request_prb_allocations[ue_rnti].get(
            earliest_req_id, 0
        )

        remaining_prbs = max(0, total_prbs_needed - prbs_allocated)

        # Calculate priority using exponential decay and remaining PRBs
        priority = self.ue_priorities[ue_rnti] + remaining_prbs * math.exp(
            -1 * time_to_deadline_s
        )
        # self.log_file.write(f"accelerate {ue_rnti} {prbs_allocated} {total_prbs_needed} {time_to_deadline_s} {priority}\n")
        # self.log_file.flush()
        return priority

    def _update_priorities(self):
        """
        Update priorities based on request timers and latency requirements.
        """
        self.log_file.write("Priority update thread started\n")
        self.log_file.flush()
        while self.running:
            try:
                current_time = time.time()
                for rnti in list(self.request_start_times.keys()):
                    # Skip if we don't have UE info yet
                    if rnti not in self.ue_info:
                        self.log_file.write(
                            f"Waiting for UE info for RNTI {rnti}\n"
                        )
                        continue

                    # Initialize priority if not exists
                    if rnti not in self.ue_priorities:
                        self._initialize_ue_priority(rnti)

                    requests = self.request_start_times[rnti]
                    if not requests:  # No requests for this UE
                        self.reset_priority(rnti)
                        continue

                    # Get UE's latency requirement
                    slo_latency = self.ue_info[rnti]["slo_latency"]
                    incentive_threshold = slo_latency / 2

                    # Find earliest request and calculate its elapsed time
                    earliest_req_id = min(requests.items(), key=lambda x: x[1])[
                        0
                    ]
                    elapsed_time_ms = (
                        current_time - requests[earliest_req_id]
                    ) * 1000

                    # Calculate priority based on current state
                    if elapsed_time_ms < incentive_threshold:
                        priority = self._calculate_incentive_priority(rnti)
                    elif elapsed_time_ms < slo_latency:
                        priority = self._calculate_accelerate_priority(rnti)
                    else:
                        priority = 10000

                    # Update priority state and scheduler
                    self.ue_priorities[rnti] = priority
                    self.set_priority(rnti, priority)

                time.sleep(0.001)  # 1ms update interval

            except Exception as e:
                self.log_file.write(f"Error updating priorities: {e}\n")
                self.log_file.flush()

    def _update_prb_history(self, rnti: int, slot: int, prbs: int):
        """
        Update PRB history and track PRB allocations for earliest active
        request.
        """
        # Initialize history for new UE if needed
        if rnti not in self.prb_history:
            self.prb_history[rnti] = {}
            # Fill with zeros for all known slots
            for s in self.slot_history:
                self.prb_history[rnti][s] = 0

        # If this is a new slot we haven't seen before
        if slot not in self.slot_history:
            self.slot_history.append(slot)

            # Add zero entries for this slot for all UEs
            for ue_rnti in self.prb_history:
                self.prb_history[ue_rnti][slot] = 0

            # Remove oldest slots if we're beyond window size
            while len(self.slot_history) > self.HISTORY_WINDOW:
                oldest_slot = self.slot_history.pop(0)
                # Remove this slot from all UE histories
                for ue_rnti in self.prb_history:
                    self.prb_history[ue_rnti].pop(oldest_slot, None)

        # Update the PRB allocation for this UE and slot
        self.prb_history[rnti][slot] = prbs

        # Update PRB allocation for earliest active request
        if rnti in self.request_start_times and self.request_start_times[rnti]:
            # Initialize allocation tracking if needed
            if rnti not in self.request_prb_allocations:
                self.request_prb_allocations[rnti] = {}

            # Get earliest request ID
            earliest_req_id = min(
                self.request_start_times[rnti].items(), key=lambda x: x[1]
            )[0]

            # Initialize or update PRB allocation for this request
            if earliest_req_id not in self.request_prb_allocations[rnti]:
                self.request_prb_allocations[rnti][earliest_req_id] = 0
            self.request_prb_allocations[rnti][earliest_req_id] += prbs

    def _handle_prb_metrics(self, rnti: int, slot: int, prbs: int):
        """
        Handle PRB allocation metrics.
        """
        # Store basic metrics
        self.current_metrics[rnti] = {
            "PRBs": prbs,
            "SLOT": slot,
        }

        # Update latest PRB allocation and slot
        self.ue_prb_status[rnti] = (slot, prbs)
        self.log_file.write(
            f"PRB received from RNTI=0x{rnti:x}, slot={slot}, prbs={prbs} at"
            f" {time.time()}\n"
        )
        self.log_file.flush()

        # Update PRB history
        self._update_prb_history(rnti, slot, prbs)

    def _handle_sr_metrics(self, rnti: int, slot: int):
        """
        Handle SR indication metrics.
        """
        self.log_file.write(
            f"SR received from RNTI=0x{rnti:x}, slot={slot} at {time.time()}\n"
        )
        self.log_file.flush()

    def _handle_bsr_metrics(self, rnti: int, slot: int, bytes: int):
        """
        Handle BSR metrics.
        """
        self.log_file.write(
            f"BSR received from RNTI=0x{rnti:x}, slot={slot}, bytes={bytes} at"
            f" {time.time()}\n"
        )
        self.log_file.flush()

        if rnti not in self.current_metrics:
            self.current_metrics[rnti] = {}
        self.current_metrics[rnti].update(
            {"BSR_BYTES": bytes, "BSR_SLOT": slot}
        )

    def __del__(self):
        """
        Ensure sockets are closed when object is destroyed.
        """
        self.stop()


def main():
    controller = TrainDataCollector()
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
