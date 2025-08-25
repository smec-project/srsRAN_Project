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
        # Application server setup (UDP)
        self.app_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Enable address/port reuse
        self.app_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.app_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.app_socket.bind(("0.0.0.0", app_port))
        # UDP doesn't need listen() or connection tracking
        self.app_port = app_port

        # RAN metrics connection setup (UDP)
        self.ran_metrics_socket = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM
        )
        self.ran_metrics_socket.setsockopt(
            socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
        )
        self.ran_metrics_socket.setsockopt(
            socket.SOL_SOCKET, socket.SO_REUSEPORT, 1
        )
        self.ran_metrics_ip = ran_metrics_ip
        self.ran_metrics_port = ran_metrics_port

        # RAN control connection setup (UDP)
        self.ran_control_socket = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM
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
        self.ue_resource_needs: Dict[str, int] = (
            {}
        )  # RNTI -> total bytes needed
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
        self.prb_history: Dict[str, Dict[int, int]] = (
            {}
        )  # RNTI -> {slot -> prbs}
        self.slot_history: list = []  # Ordered list of slots we've seen

        # Track allocated PRBs for each request
        self.request_prb_allocations: Dict[str, Dict[int, int]] = (
            {}
        )  # RNTI -> {request_id -> total_allocated_prbs}

        # Open log file first
        self.log_file = open("controller.txt", "w")

        # Then start priority update thread
        self.priority_thread = threading.Thread(target=self._update_priorities)
        self.priority_thread.start()

    def start(self):
        """
        Start the controller and all its connections.
        """
        self.running = True

        # Bind RAN metrics socket (UDP control socket doesn't need connect)
        try:
            self.ran_metrics_socket.bind(
                (self.ran_metrics_ip, self.ran_metrics_port)
            )
            self.log_file.write(f"RAN metrics UDP server listening on {self.ran_metrics_ip}:{self.ran_metrics_port}\n")
            self.log_file.write(f"RAN control UDP sender initialized (target: {self.ran_control_ip}:{self.ran_control_port})\n")
            self.log_file.flush()
        except Exception as e:
            self.log_file.write(f"Failed to setup RAN services: {e}\n")
            self.log_file.flush()
            return False

        # Start threads for different functionalities
        threading.Thread(
            target=self._handle_app_messages, daemon=True
        ).start()
        threading.Thread(target=self._handle_ran_metrics, daemon=True).start()

        return True

    def _handle_app_messages(self):
        """
        Handle incoming UDP application messages.
        Message format: 5 integers (rnti, request_index, request_size, slo_ms, start_or_end)
        """
        while self.running:
            try:
                data, addr = self.app_socket.recvfrom(1024)
                if not data:
                    continue
                
                # Expect 20 bytes (5 x 32-bit integers)
                if len(data) != 20:
                    self.log_file.write(f"Invalid message size: {len(data)} bytes, expected 20\n")
                    self.log_file.flush()
                    continue
                
                # Unpack: rnti, request_index, request_size, slo_ms, start_or_end
                rnti, request_index, request_size, slo_ms, start_or_end = struct.unpack('IIIII', data)
                
                # Convert RNTI to string for compatibility with existing code
                rnti_str = f"{rnti:04x}"
                
                self.log_file.write(
                    f"UDP message from {addr}: RNTI=0x{rnti_str}, req_idx={request_index}, "
                    f"size={request_size}, slo={slo_ms}ms, start_end={start_or_end}\n"
                )
                self.log_file.flush()
                
                # Process the message
                self._process_udp_app_message(rnti_str, request_index, request_size, slo_ms, start_or_end)
                
            except Exception as e:
                self.log_file.write(f"Error receiving UDP application message: {e}\n")
                self.log_file.flush()

    def _process_udp_app_message(self, rnti_str: str, request_index: int, request_size: int, slo_ms: int, start_or_end: int):
        """
        Process UDP application message.
        
        Args:
            rnti_str: RNTI as hex string
            request_index: Request sequence number
            request_size: Request size in bytes  
            slo_ms: SLO latency requirement in milliseconds
            start_or_end: 0 for start, 1 for end
        """
        try:
            if start_or_end == 0:  # Request START
                # Auto-register UE if not exists
                if rnti_str not in self.ue_info:
                    self.ue_info[rnti_str] = {
                        "app_id": "udp_app",
                        "ue_idx": "auto", 
                        "latency_req": float(slo_ms),
                        "request_size": request_size,
                    }
                    self.request_sequences[rnti_str] = []
                    self.ue_resource_needs[rnti_str] = 0
                    self.ue_pending_requests[rnti_str] = {}
                    self.request_start_times[rnti_str] = {}
                    
                    self.log_file.write(
                        f"Auto-registered UE - RNTI: {rnti_str}, SLO: {slo_ms}ms, Size: {request_size} bytes\n"
                    )
                    self.log_file.flush()
                
                # Start new request
                self.ue_resource_needs[rnti_str] += request_size
                self.ue_pending_requests[rnti_str][request_index] = request_size
                
                if rnti_str not in self.request_start_times:
                    self.request_start_times[rnti_str] = {}
                self.request_start_times[rnti_str][request_index] = time.time()
                
                self.log_file.write(
                    f"Request {request_index} from RNTI {rnti_str} started at {time.time()}\n"
                )
                self.log_file.flush()
                
            elif start_or_end == 1:  # Request END
                # Calculate completion time
                if (
                    rnti_str in self.request_start_times
                    and request_index in self.request_start_times[rnti_str]
                ):
                    start_time = self.request_start_times[rnti_str][request_index]
                    elapsed_time_ms = (time.time() - start_time) * 1000
                    self.log_file.write(
                        f"Request {request_index} from RNTI {rnti_str} completed in {elapsed_time_ms:.2f}ms at {time.time()}\n"
                    )
                    self.log_file.flush()
                    del self.request_start_times[rnti_str][request_index]
                
                # Handle resource cleanup
                if (
                    rnti_str in self.ue_pending_requests
                    and request_index in self.ue_pending_requests[rnti_str]
                ):
                    completed_size = self.ue_pending_requests[rnti_str][request_index]
                    self.ue_resource_needs[rnti_str] -= completed_size
                    del self.ue_pending_requests[rnti_str][request_index]
                
            else:
                self.log_file.write(f"Invalid start_or_end value: {start_or_end}\n")
                self.log_file.flush()
                
        except Exception as e:
            self.log_file.write(f"Error processing UDP app message: {e}\n")
            self.log_file.flush()

    def _handle_ran_metrics(self):
        """
        Receive and process RAN metrics (UDP binary format).
        """
        while self.running:
            try:
                data, addr = self.ran_metrics_socket.recvfrom(1024)
                if not data:
                    continue

                # Process binary messages (16 bytes each: 4 x 32-bit integers)
                message_size = 16
                offset = 0
                
                while offset + message_size <= len(data):
                    # Unpack: msg_type, rnti, field1, field2
                    msg_type, rnti, field1, field2 = struct.unpack('IIII', data[offset:offset + message_size])
                    
                    if msg_type == 0:  # PRB
                        self._handle_prb_metrics(rnti, field1, field2)
                    elif msg_type == 1:  # SR
                        self._handle_sr_metrics(rnti, field1)
                    elif msg_type == 2:  # BSR
                        self._handle_bsr_metrics(rnti, field1, field2)
                    else:
                        self.log_file.write(f"Unknown metrics type: {msg_type}\n")
                        self.log_file.flush()
                    
                    offset += message_size

            except Exception as e:
                self.log_file.write(f"Error receiving RAN metrics: {e}\n")
                self.log_file.flush()

    def set_priority(self, rnti: str, priority: float):
        """
        Send priority update to RAN via UDP.
        """
        try:
            # Convert string RNTI to integer for binary format compatibility
            rnti_int = int(rnti, 16)  # Convert hex string to int
            
            # Pack message: RNTI as int, priority as double, reset as bool
            msg = struct.pack("=IdB", rnti_int, priority, False)

            # Send via UDP to target address
            self.ran_control_socket.sendto(
                msg,
                (self.ran_control_ip, self.ran_control_port)
            )
            return True
        except Exception as e:
            self.log_file.write(f"Failed to send priority update: {e}\n")
            self.log_file.flush()
            return False

    def reset_priority(self, rnti: str):
        """
        Reset priority for a specific RNTI via UDP.
        """
        try:
            # Reset priority state
            if rnti in self.ue_priorities:
                self.ue_priorities[rnti] = 0.0

            # Convert string RNTI to integer for binary format compatibility
            rnti_int = int(rnti, 16)  # Convert hex string to int
            
            # Pack reset message: RNTI as int, 0.0 priority, reset=True
            msg = struct.pack("=IdB", rnti_int, 0.0, True)
            
            # Send via UDP to target address
            self.ran_control_socket.sendto(
                msg,
                (self.ran_control_ip, self.ran_control_port)
            )
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

        # Close server sockets (UDP app doesn't maintain connections)
        # Close sockets (UDP sockets don't need shutdown)
        sockets_to_close = [
            self.app_socket,
            self.ran_metrics_socket,
            self.ran_control_socket
        ]
        
        for sock in sockets_to_close:
            if sock:
                try:
                    # Only TCP sockets need shutdown, UDP can be closed directly
                    if sock == self.app_socket:  # TCP socket
                        sock.shutdown(socket.SHUT_RDWR)
                    sock.close()
                except:
                    pass

    def _initialize_ue_priority(self, rnti: str):
        """
        Initialize priority for a new UE.
        """
        self.ue_priorities[rnti] = 0.0

    def _update_priorities(self):
        """
        Update priorities based on request timers and latency requirements.
        """
        self.log_file.write("Priority update thread started\n")
        self.log_file.flush()

        # Track reset state for each UE
        ue_reset_state = {}  # RNTI -> whether priority is reset

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
                        ue_reset_state[rnti] = False

                    requests = self.request_start_times[rnti]
                    if not requests:  # No requests for this UE
                        # Only reset if not already reset
                        if not ue_reset_state.get(rnti, False):
                            self.reset_priority(rnti)
                            ue_reset_state[rnti] = True
                            self.log_file.write(
                                f"No requests for RNTI {rnti}, resetting"
                                " priority\n"
                            )
                            self.log_file.flush()
                        continue

                    # If we have requests, clear reset state
                    ue_reset_state[rnti] = False

                    # Get UE's latency requirement and calculate priority
                    latency_req = self.ue_info[rnti]["latency_req"]
                    elapsed_time_ms = (
                        current_time
                        - requests[min(requests.items(), key=lambda x: x[1])[0]]
                    ) * 1000
                    deadline_ms = latency_req - elapsed_time_ms
                    remaining_seconds = deadline_ms / 1000.0
                    priority = 1.0 / (
                        remaining_seconds * remaining_seconds + 1e-6
                    )

                    # Update priority state and scheduler
                    self.ue_priorities[rnti] = priority
                    self.set_priority(rnti, priority)
                    self.log_file.write(
                        f"Updated priority for RNTI {rnti} to {priority} at"
                        f" {time.time()}\n"
                    )
                    self.log_file.flush()

                time.sleep(0.001)  # 1ms update interval

            except Exception as e:
                self.log_file.write(f"Error updating priorities: {e}\n")
                self.log_file.flush()

    def _update_prb_history(self, rnti: str, slot: int, prbs: int):
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

    def _handle_prb_metrics(self, rnti, prbs, slot):
        """
        Handle PRB allocation metrics.
        """
        # Convert RNTI to string for compatibility
        rnti_str = f"{rnti:04x}"  # Convert to 4-digit hex string
        
        # Store basic metrics
        self.current_metrics[rnti_str] = {
            "PRBs": prbs,
            "SLOT": slot,
        }

        # Update latest PRB allocation and slot
        self.ue_prb_status[rnti_str] = (slot, prbs)
        self.log_file.write(
            f"PRB received from RNTI=0x{rnti_str}, slot={slot}, prbs={prbs} at"
            f" {time.time()}\n"
        )
        self.log_file.flush()

        # Update PRB history
        self._update_prb_history(rnti_str, slot, prbs)

    def _handle_sr_metrics(self, rnti, slot):
        """
        Handle SR indication metrics.
        """
        # Convert RNTI to string for compatibility
        rnti_str = f"{rnti:04x}"  # Convert to 4-digit hex string
        
        self.log_file.write(
            f"SR received from RNTI=0x{rnti_str}, slot={slot} at {time.time()}\n"
        )
        self.log_file.flush()

    def _handle_bsr_metrics(self, rnti, bytes_val, slot):
        """
        Handle BSR metrics.
        """
        # Convert RNTI to string for compatibility
        rnti_str = f"{rnti:04x}"  # Convert to 4-digit hex string
        
        self.log_file.write(
            f"bsr received from RNTI=0x{rnti_str}, slot={slot}, bytes={bytes_val} at"
            f" {time.time()}\n"
        )
        self.log_file.flush()

        if rnti_str not in self.current_metrics:
            self.current_metrics[rnti_str] = {}
        self.current_metrics[rnti_str].update(
            {"BSR_BYTES": bytes_val, "BSR_SLOT": slot}
        )

    def __del__(self):
        """
        Ensure sockets are closed when object is destroyed.
        """
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
