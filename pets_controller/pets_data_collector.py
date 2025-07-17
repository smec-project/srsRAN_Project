import socket
import struct
import threading
import time
import math
from typing import Dict, Optional


class TuttiController:
    def __init__(
        self,
        app_port: int = 5557,  # Port to receive application messages
        ran_metrics_ip: str = "127.0.0.1",
        ran_metrics_port: int = 5556,  # Port to receive RAN metrics
        ran_control_ip: str = "127.0.0.1",
        ran_control_port: int = 5555,  # Port to send priority updates
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
        self.app_requirements: Dict[str, Dict[str, dict]] = (
            {}
        )  # UE_IDX -> {app_id -> requirements}

        # Track resource requirements and pending requests for each UE
        self.ue_resource_needs: Dict[str, int] = {}  # RNTI -> total bytes needed
        self.ue_pending_requests: Dict[str, Dict[int, int]] = (
            {}
        )  # RNTI -> {request_id -> bytes}

        # Track latest PRB allocation and slot for each UE
        self.ue_prb_status: Dict[str, tuple] = {}  # RNTI -> (slot, prbs)

        # Change request_timers to store start timestamps
        self.request_start_times: Dict[str, Dict[int, float]] = (
            {}
        )  # RNTI -> {request_id -> start_timestamp}

        # Track UE priority states
        self.ue_priorities: Dict[str, float] = {}  # RNTI -> current_priority

        # Track PRB allocation history
        self.HISTORY_WINDOW = 100  # Keep last 100 slot records
        self.prb_history: Dict[str, Dict[int, int]] = {}  # RNTI -> {slot -> prbs}
        self.slot_history: list = []  # Ordered list of slots we've seen

        # Track allocated PRBs for each request
        self.request_prb_allocations: Dict[str, Dict[int, int]] = (
            {}
        )  # RNTI -> {request_id -> total_allocated_prbs}

        # Open log file first
        self.log_file = open("controller.txt", "w")

        # Then start priority update thread
        # self.priority_thread = threading.Thread(target=self._update_priorities)
        # self.priority_thread.start()

        # Add mapping between UE_IDX and RNTI
        self.ue_idx_to_rnti = {}  # UE_IDX -> RNTI
        self.rnti_to_ue_idx = {}  # RNTI -> UE_IDX

    def start(self):
        """Start the controller and all its connections"""
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
        threading.Thread(target=self._handle_app_connections, daemon=True).start()
        threading.Thread(target=self._handle_ran_metrics, daemon=True).start()

        return True

    def _handle_app_connections(self):
        """Handle incoming application connections and messages"""
        while self.running:
            try:
                conn, addr = self.app_socket.accept()
                self.log_file.write(f"New application connected from {addr}\n")
                self.log_file.flush()
                self.app_connections[addr[0]] = conn
                threading.Thread(
                    target=self._handle_app_messages, args=(conn, addr), daemon=True
                ).start()
            except Exception as e:
                self.log_file.write(f"Error accepting application connection: {e}\n")
                self.log_file.flush()

    def _handle_app_messages(self, conn: socket.socket, addr):
        """Handle messages from a specific application connection"""
        try:
            # First message should be app registration with app_id
            data = conn.recv(1024)
            if not data:
                return

            app_id = data.decode("utf-8").strip()
            self.app_connections[app_id] = conn
            self.log_file.write(f"Application {app_id} registered from {addr}\n")
            self.log_file.flush()

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
                        self.log_file.write(
                            f"New UE registered - RNTI: {rnti}, UE_IDX: {ue_idx}, "
                            f"Latency Req: {latency_req}ms, Size: {request_size} bytes\n"
                        )
                        self.log_file.flush()
                        # Initialize resource tracking for new UE
                        self.ue_resource_needs[rnti] = 0
                        self.ue_pending_requests[rnti] = {}
                        self.request_start_times[rnti] = {}

                    elif msg_type == "Start":
                        # Format: "Start|rnti|seq_number"
                        _, rnti, seq_num = msg_parts
                        current_time = time.time()
                        self.log_file.write(
                            f"Request {seq_num} from RNTI {rnti} start at {current_time}\n"
                        )
                        self.log_file.flush()

                    elif msg_type == "REQUEST":
                        # Format: "REQUEST|RNTI|SEQ_NUM"
                        _, rnti, seq_num = msg_parts
                        seq_num = int(seq_num)

                        if rnti in self.ue_info:
                            # Calculate request size in bytes
                            request_size = self.ue_info[rnti]["request_size"]

                            # Update resource needs and pending requests
                            self.ue_resource_needs[rnti] += request_size
                            self.ue_pending_requests[rnti][seq_num] = request_size

                            # Store request start time
                            if rnti not in self.request_start_times:
                                self.request_start_times[rnti] = {}
                            self.request_start_times[rnti][seq_num] = time.time()
                        else:
                            self.log_file.write(
                                f"Warning: Request for unknown RNTI {rnti}\n"
                            )
                            self.log_file.flush()

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
                            self.log_file.write(
                                f"Request {seq_num} from RNTI {rnti} completed in {elapsed_time_ms:.2f}ms at {time.time()}\n"
                            )
                            self.log_file.flush()
                            del self.request_start_times[rnti][seq_num]

                        # Handle resource tracking
                        if (
                            rnti in self.ue_pending_requests
                            and seq_num in self.ue_pending_requests[rnti]
                        ):
                            completed_size = self.ue_pending_requests[rnti][seq_num]
                            self.ue_resource_needs[rnti] -= completed_size
                            del self.ue_pending_requests[rnti][seq_num]

                        if (
                            rnti not in self.request_start_times
                            and rnti not in self.ue_pending_requests
                        ):
                            self.log_file.write(
                                f"Warning: Completion for unknown RNTI {rnti}\n"
                            )
                            self.log_file.flush()

                except Exception as e:
                    self.log_file.write(f"Error processing application message: {e}\n")
                    self.log_file.flush()
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
                self.log_file.write(f"Error receiving RAN metrics: {e}\n")
                self.log_file.flush()

    def set_priority(self, rnti: str, priority: float):
        """Send priority update to RAN"""
        try:
            # Format RNTI string and pack message in the correct format
            rnti_str = f"{rnti:<4}".encode("ascii")  # Left align, space pad to 4 chars
            msg = struct.pack("=5sdb", rnti_str, priority, False)

            self.ran_control_socket.send(msg)
            return True
        except Exception as e:
            self.log_file.write(f"Failed to send priority update: {e}\n")
            self.log_file.flush()
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
            self.log_file.write(f"Failed to reset priority: {e}\n")
            self.log_file.flush()
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

    def _initialize_ue_priority(self, rnti: str):
        """Initialize priority for a new UE"""
        self.ue_priorities[rnti] = 0.0

    def _calculate_incentive_priority(self, ue_rnti: str) -> float:
        """Calculate priority for incentive mode (first half of latency requirement)"""
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
            latency_req = self.ue_info[rnti]["latency_req"]
            available_ttis = latency_req / TTI_DURATION
            prbs_per_tti = (total_prbs + int(available_ttis) - 1) // int(available_ttis)
            ue_prb_requirements[rnti] = (total_prbs, prbs_per_tti)

        # Get the latest slot from history
        if not self.slot_history:
            return DEFAULT_PRIORITY_OFFSET
        latest_slot = self.slot_history[-1]

        # Calculate priority adjustments based on PRB differences
        priority_adjustments = {}
        for rnti, (_, required_prbs_per_tti) in ue_prb_requirements.items():
            if rnti not in self.prb_history:
                self.log_file.write(f"Warning: No PRB history for RNTI {rnti}\n")
                self.log_file.flush()
                return DEFAULT_PRIORITY_OFFSET
            actual_prbs = self.prb_history[rnti][latest_slot]
            prb_difference = actual_prbs - required_prbs_per_tti
            current_offset = self.ue_priorities.get(rnti, DEFAULT_PRIORITY_OFFSET)
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
                    abs(actual_prbs - ue_prb_requirements[ue_rnti][1]) / actual_prbs,
                )
            else:
                current_offset = self.ue_priorities[ue_rnti] + abs(
                    actual_prbs - ue_prb_requirements[ue_rnti][1]
                )
        # self.log_file.write(f"incentive {ue_rnti} {actual_prbs} {ue_prb_requirements[ue_rnti][1]} {current_offset}\n")
        # self.log_file.flush()  # Ensure immediate write
        return current_offset

    def _calculate_accelerate_priority(self, ue_rnti: str) -> float:
        """Calculate priority for accelerate mode (second half of latency requirement)"""
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
        latency_req_ms = self.ue_info[ue_rnti]["latency_req"]

        # Calculate time to deadline in seconds
        time_to_deadline_s = (latency_req_ms - elapsed_time_ms) * MS_TO_S

        # Calculate remaining PRBs needed
        total_prbs_needed = (
            self.ue_pending_requests[ue_rnti][earliest_req_id] + BYTES_PER_PRB - 1
        ) // BYTES_PER_PRB

        # Safely get allocated PRBs
        if ue_rnti not in self.request_prb_allocations:
            self.request_prb_allocations[ue_rnti] = {}
        prbs_allocated = self.request_prb_allocations[ue_rnti].get(earliest_req_id, 0)

        remaining_prbs = max(0, total_prbs_needed - prbs_allocated)

        # Calculate priority using exponential decay and remaining PRBs
        priority = self.ue_priorities[ue_rnti] + remaining_prbs * math.exp(
            -1 * time_to_deadline_s
        )
        # self.log_file.write(f"accelerate {ue_rnti} {prbs_allocated} {total_prbs_needed} {time_to_deadline_s} {priority}\n")
        # self.log_file.flush()
        return priority

    def _update_priorities(self):
        """Update priorities based on request timers and latency requirements"""
        self.log_file.write("Priority update thread started\n")
        self.log_file.flush()
        while self.running:
            try:
                current_time = time.time()
                for rnti in list(self.request_start_times.keys()):
                    # Skip if we don't have UE info yet
                    if rnti not in self.ue_info:
                        self.log_file.write(f"Waiting for UE info for RNTI {rnti}\n")
                        continue

                    # Initialize priority if not exists
                    if rnti not in self.ue_priorities:
                        self._initialize_ue_priority(rnti)

                    requests = self.request_start_times[rnti]
                    if not requests:  # No requests for this UE
                        self.reset_priority(rnti)
                        continue

                    # Get UE's latency requirement
                    latency_req = self.ue_info[rnti]["latency_req"]
                    incentive_threshold = latency_req / 2

                    # Find earliest request and calculate its elapsed time
                    earliest_req_id = min(requests.items(), key=lambda x: x[1])[0]
                    elapsed_time_ms = (current_time - requests[earliest_req_id]) * 1000

                    # Calculate priority based on current state
                    if elapsed_time_ms < incentive_threshold:
                        priority = self._calculate_incentive_priority(rnti)
                    elif elapsed_time_ms < latency_req:
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

    def _update_prb_history(self, rnti: str, slot: int, prbs: int):
        """Update PRB history and track PRB allocations for earliest active request"""
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

    def _handle_prb_metrics(self, values):
        """Handle PRB allocation metrics"""
        # Keep RNTI as string
        rnti = values["RNTI"][-4:]  # Just take last 4 chars
        ue_idx = values["UE_IDX"]
        slot = int(values["SLOT"])
        prbs = int(values["PRBs"])

        # Update UE_IDX <-> RNTI mapping
        self.ue_idx_to_rnti[ue_idx] = rnti
        self.rnti_to_ue_idx[rnti] = ue_idx

        # Store basic metrics
        self.current_metrics[rnti] = {"UE_IDX": ue_idx, "PRBs": prbs, "SLOT": slot}

        # Update latest PRB allocation and slot
        self.ue_prb_status[rnti] = (slot, prbs)
        self.log_file.write(
            f"PRB received from RNTI=0x{rnti}, slot={slot}, prbs={prbs} at {time.time()}\n"
        )
        self.log_file.flush()

        # Update PRB history
        self._update_prb_history(rnti, slot, prbs)

    def _handle_sr_metrics(self, values):
        """Handle SR indication metrics"""
        rnti = values["RNTI"][-4:]  # Just take last 4 chars
        slot = int(values["SLOT"])
        self.log_file.write(
            f"SR received from RNTI=0x{rnti}, slot={slot} at {time.time()}\n"
        )
        self.log_file.flush()

    def _handle_bsr_metrics(self, values):
        """Handle BSR metrics"""
        rnti = values["RNTI"][-4:]  # Just take last 4 chars
        ue_idx = values["UE_IDX"]
        bytes = int(values["BYTES"])
        slot = int(values["SLOT"])

        self.log_file.write(
            f"bsr received from RNTI=0x{rnti}, slot={slot}, bytes={bytes} at {time.time()}\n"
        )
        self.log_file.flush()

        if rnti not in self.current_metrics:
            self.current_metrics[rnti] = {}
        self.current_metrics[rnti].update(
            {"UE_IDX": ue_idx, "BSR_BYTES": bytes, "BSR_SLOT": slot}
        )

    def __del__(self):
        """Ensure sockets are closed when object is destroyed"""
        self.stop()


def main():
    controller = TuttiController()
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
