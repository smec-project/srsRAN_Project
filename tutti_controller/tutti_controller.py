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
        self.app_socket.bind(("0.0.0.0", app_port))
        self.app_socket.listen(5)
        self.app_connections: Dict[str, socket.socket] = {}
        
        # RAN metrics connection setup
        self.ran_metrics_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ran_metrics_ip = ran_metrics_ip
        self.ran_metrics_port = ran_metrics_port
        
        # RAN control connection setup
        self.ran_control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ran_control_ip = ran_control_ip
        self.ran_control_port = ran_control_port
        
        # State tracking
        self.running = False
        self.current_metrics: Dict[str, dict] = {}  # RNTI (str) -> metrics
        self.ue_info: Dict[str, dict] = {}  # RNTI (str) -> {app_id, latency_req, request_size, ue_idx}
        self.request_sequences: Dict[str, list] = {}  # RNTI (str) -> list of sequence numbers
        self.app_requirements: Dict[str, Dict[str, dict]] = {}  # UE_IDX -> {app_id -> requirements}
        
        # Track resource requirements and pending requests for each UE
        self.ue_resource_needs: Dict[str, int] = {}  # RNTI -> total bytes needed
        self.ue_pending_requests: Dict[str, Dict[int, int]] = {}  # RNTI -> {request_id -> bytes}
        
        # Track latest PRB allocation and slot for each UE
        self.ue_prb_status: Dict[str, tuple] = {}  # RNTI -> (slot, prbs)
        
        # Track request timers (elapsed time in ms)
        self.request_timers: Dict[str, Dict[int, int]] = {}  # RNTI -> {request_id -> elapsed_ms}
        
        # Track UE priority states
        self.ue_priorities: Dict[str, float] = {}  # RNTI -> current_priority
        
        # Track PRB allocation history
        self.HISTORY_WINDOW = 100  # Keep last 100 slot records
        self.prb_history: Dict[str, Dict[int, int]] = {}  # RNTI -> {slot -> prbs}
        self.slot_history: list = []  # Ordered list of slots we've seen
        
        # Track allocated PRBs for each request
        self.request_prb_allocations: Dict[str, Dict[int, int]] = {}  # RNTI -> {request_id -> total_allocated_prbs}
        
        # Start timer update thread
        threading.Thread(target=self._update_request_timers, daemon=True).start()
        
        # Start priority update thread
        self.priority_thread = threading.Thread(target=self._update_priorities)
        self.priority_thread.start()

    def start(self):
        """Start the controller and all its connections"""
        self.running = True
        
        # Connect to RAN services
        try:
            self.ran_metrics_socket.connect((self.ran_metrics_ip, self.ran_metrics_port))
            self.ran_control_socket.connect((self.ran_control_ip, self.ran_control_port))
        except Exception as e:
            print(f"Failed to connect to RAN services: {e}")
            return False

        # Start threads for different functionalities
        threading.Thread(target=self._handle_app_connections, daemon=True).start()
        threading.Thread(target=self._handle_ran_metrics, daemon=True).start()
        # threading.Thread(target=self._process_and_update, daemon=True).start()
        
        return True

    def _handle_app_connections(self):
        """Handle incoming application connections and messages"""
        while self.running:
            try:
                conn, addr = self.app_socket.accept()
                print(f"New application connected from {addr}")
                self.app_connections[addr[0]] = conn
                threading.Thread(
                    target=self._handle_app_messages,
                    args=(conn, addr),
                    daemon=True
                ).start()
            except Exception as e:
                print(f"Error accepting application connection: {e}")

    def _handle_app_messages(self, conn: socket.socket, addr):
        """Handle messages from a specific application connection"""
        try:
            # First message should be app registration with app_id
            data = conn.recv(1024)
            if not data:
                return
            
            app_id = data.decode('utf-8').strip()
            self.app_connections[app_id] = conn
            print(f"Application {app_id} registered from {addr}")
            
            while self.running:
                try:
                    data = conn.recv(1024)
                    if not data:
                        break
                    
                    message = data.decode('utf-8').strip()
                    msg_parts = message.split('|')
                    msg_type = msg_parts[0]

                    if msg_type == "NEW_UE":
                        # Format: "NEW_UE|RNTI|UE_IDX|LATENCY_REQ|REQUEST_SIZE"
                        _, rnti, ue_idx, latency_req, request_size = msg_parts
                        # Store RNTI as string
                        self.ue_info[rnti] = {
                            'app_id': app_id,
                            'ue_idx': ue_idx,
                            'latency_req': float(latency_req),
                            'request_size': int(request_size)
                        }
                        self.request_sequences[rnti] = []
                        print(f"New UE registered - RNTI: {rnti}, UE_IDX: {ue_idx}, "
                              f"Latency Req: {latency_req}ms, Size: {request_size} bytes")
                        # Initialize resource tracking for new UE
                        self.ue_resource_needs[rnti] = 0
                        self.ue_pending_requests[rnti] = {}
                        self.request_timers[rnti] = {}

                    elif msg_type == "REQUEST":
                        # Format: "REQUEST|RNTI|SEQ_NUM"
                        _, rnti, seq_num = msg_parts
                        seq_num = int(seq_num)
                        
                        if rnti in self.ue_info:
                            # Calculate request size in bytes
                            request_size = self.ue_info[rnti]['request_size']
                            
                            # Update resource needs and pending requests
                            self.ue_resource_needs[rnti] += request_size
                            self.ue_pending_requests[rnti][seq_num] = request_size

                            # print(f"New request {seq_num} for RNTI {rnti}:")
                            # print(f"  Request size: {request_size} bytes")
                            # print(f"  Total resource need: {self.ue_resource_needs[rnti]} bytes")
                            # print(f"  Pending requests: {len(self.ue_pending_requests[rnti])}")
                            # Initialize timer for new request
                            self.request_timers[rnti][seq_num] = 0
                        else:
                            print(f"Warning: Request for unknown RNTI {rnti}")

                    elif msg_type == "DONE":
                        # Format: "DONE|RNTI|SEQ_NUM"
                        _, rnti, seq_num = msg_parts
                        seq_num = int(seq_num)
                        
                        # Handle request timer
                        if rnti in self.request_timers and seq_num in self.request_timers[rnti]:
                            # final_time = self.request_timers[rnti][seq_num]
                            # print(f"Request {seq_num} from RNTI {rnti} completed in {final_time}ms")
                            del self.request_timers[rnti][seq_num]
                        
                        # Handle resource tracking
                        if rnti in self.ue_pending_requests and seq_num in self.ue_pending_requests[rnti]:
                            completed_size = self.ue_pending_requests[rnti][seq_num]
                            self.ue_resource_needs[rnti] -= completed_size
                            del self.ue_pending_requests[rnti][seq_num]
                            # print(f"Request {seq_num} completed for RNTI {rnti}:")
                            # print(f"  Freed resources: {completed_size} bytes")
                            # print(f"  Remaining resource need: {self.ue_resource_needs[rnti]} bytes")
                            # print(f"  Remaining requests: {len(self.ue_pending_requests[rnti])}")
                        
                        if rnti not in self.request_timers and rnti not in self.ue_pending_requests:
                            print(f"Warning: Completion for unknown RNTI {rnti}")

                except Exception as e:
                    print(f"Error processing application message: {e}")
                    break
                    
        finally:
            if app_id in self.app_connections:
                del self.app_connections[app_id]
            conn.close()

    def _handle_ran_metrics(self):
        """Receive and process RAN metrics"""
        while self.running:
            try:
                data = self.ran_metrics_socket.recv(1024).decode('utf-8')
                if not data:
                    continue
                
                # Parse the metrics string
                metrics = {}
                for line in data.strip().split('\n'):
                    values = dict(item.split('=') for item in line.split(','))
                    # Keep RNTI as string
                    rnti = values['RNTI'][-4:]  # Just take last 4 chars
                    slot = int(values['SLOT'])
                    prbs = int(values['PRBs'])
                    
                    # Store basic metrics
                    metrics[rnti] = {
                        'UE_IDX': values['UE_IDX'],
                        'PRBs': prbs,
                        'SLOT': slot
                    }
                    
                    # Update latest PRB allocation and slot
                    self.ue_prb_status[rnti] = (slot, prbs)
                    # print(f"Updated PRB status for RNTI {rnti}: Slot {slot}, PRBs {prbs}")
                
                self.current_metrics = metrics
                
                # Update PRB history when receiving new metrics
                for rnti, metrics in self.current_metrics.items():
                    print(f"Received metrics for RNTI {rnti}: {metrics}")
                    if 'SLOT' in metrics and 'PRBs' in metrics:
                        self._update_prb_history(rnti, metrics['SLOT'], metrics['PRBs'])
                
            except Exception as e:
                print(f"Error receiving RAN metrics: {e}")

    def _process_and_update(self):
        """Process metrics and requirements, update priorities"""
        while self.running:
            try:
                # Process each UE's requirements and current state
                for rnti, info in self.ue_info.items():
                    if rnti in self.current_metrics:
                        # Get current resource requirements
                        resource_need = self.ue_resource_needs.get(rnti, 0)
                        pending_count = len(self.ue_pending_requests.get(rnti, {}))
                        
                        # Get latest PRB allocation
                        slot, prbs = self.ue_prb_status.get(rnti, (0, 0))
                        
                        print(f"UE RNTI {rnti} (IDX {info['ue_idx']}) Status:")
                        print(f"  Resource Need: {resource_need} bytes")
                        print(f"  Pending Requests: {pending_count}")
                        print(f"  Current PRBs: {prbs}")
                        print(f"  Current Slot: {slot}")
                        print(f"  Latency Requirement: {info['latency_req']}ms")
                
                time.sleep(1)
            except Exception as e:
                print(f"Error in processing loop: {e}")

    def set_priority(self, rnti: str, priority: float):
        """Send priority update to RAN"""
        try:
            # Format RNTI string and pack message in the correct format
            rnti_str = f"{rnti:<4}".encode('ascii')  # Left align, space pad to 4 chars
            msg = struct.pack('=5sdb', rnti_str, priority, False)
            
            # Debug output
            print(f"Setting priority for RNTI {rnti} to {priority}")
            print(f"Message bytes: {[hex(x) for x in msg]}")
            
            self.ran_control_socket.send(msg)
            return True
        except Exception as e:
            print(f"Failed to send priority update: {e}")
            return False

    def reset_priority(self, rnti: str):
        """Reset priority for a specific RNTI"""
        try:
            # Reset priority state
            if rnti in self.ue_priorities:
                self.ue_priorities[rnti] = 0.0
            
            # Reset in scheduler
            rnti_str = f"{rnti:<4}".encode('ascii')
            msg = struct.pack('=5sdb', rnti_str, 0.0, True)
            self.ran_control_socket.send(msg)
            print(f"Reset priority for RNTI {rnti}")
            return True
        except Exception as e:
            print(f"Failed to reset priority: {e}")
            return False

    def stop(self):
        """Stop the controller and clean up connections"""
        self.running = False
        
        # Close all application connections
        for conn in self.app_connections.values():
            conn.close()
        
        # Close server sockets
        self.app_socket.close()
        self.ran_metrics_socket.close()
        self.ran_control_socket.close()

    def _update_request_timers(self):
        """Update timers for all active requests"""
        print("Timer update thread started")  # Debug: Confirm thread start
        update_count = 0
        while self.running:
            try:
                for rnti in list(self.request_timers.keys()):
                    for req_id in list(self.request_timers[rnti].keys()):
                        self.request_timers[rnti][req_id] += 1
                        if update_count % 1000 == 0:  # Print every 1000 updates
                            print(f"Timer update: RNTI {rnti}, Request {req_id}, Value {self.request_timers[rnti][req_id]}ms")
                update_count += 1
                time.sleep(0.001)
            except Exception as e:
                print(f"Error updating request timers: {e}")

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
        for rnti in self.request_timers.keys():
            if not self.request_timers[rnti]:
                continue
                
            earliest_req_id = min(self.request_timers[rnti].items(), key=lambda x: x[1])[0]
            request_size = self.ue_pending_requests[rnti][earliest_req_id]
            total_prbs = (request_size + BYTES_PER_PRB - 1) // BYTES_PER_PRB
            latency_req = self.ue_info[rnti]['latency_req']
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
                current_offset = self.ue_priorities[ue_rnti] + max(DEFAULT_PRIORITY_OFFSET, abs(actual_prbs - ue_prb_requirements[ue_rnti][0])/actual_prbs)
            else:
                current_offset = self.ue_priorities[ue_rnti] + DEFAULT_PRIORITY_OFFSET
        return current_offset

    def _calculate_accelerate_priority(self, ue_rnti: str) -> float:
        """Calculate priority for accelerate mode (second half of latency requirement)"""
        # Constants
        BYTES_PER_PRB = 90
        MS_TO_S = 0.001  # Convert ms to s
        
        # Get current request info
        if not self.request_timers[ue_rnti]:
            return 0.0
            
        # Get earliest request and its timer
        earliest_req_id = min(self.request_timers[ue_rnti].items(), key=lambda x: x[1])[0]
        elapsed_time_ms = self.request_timers[ue_rnti][earliest_req_id]
        latency_req_ms = self.ue_info[ue_rnti]['latency_req']
        
        # Calculate time to deadline in seconds
        time_to_deadline_s = (latency_req_ms - elapsed_time_ms) * MS_TO_S
        
        # Calculate remaining PRBs needed
        total_prbs_needed = (self.ue_pending_requests[ue_rnti][earliest_req_id] + BYTES_PER_PRB - 1) // BYTES_PER_PRB
        prbs_allocated = self.request_prb_allocations[ue_rnti].get(earliest_req_id, 0)
        remaining_prbs = max(0, total_prbs_needed - prbs_allocated)
        
        # Calculate priority using exponential decay and remaining PRBs
        priority = remaining_prbs * math.exp(-1 * time_to_deadline_s)
        print(f"Accelerate calculation for RNTI {ue_rnti}:")
        print(f"  Time to deadline: {time_to_deadline_s:.3f}s")
        print(f"  Remaining PRBs: {remaining_prbs}")
        print(f"  Priority: {priority}")
        return priority

    def _update_priorities(self):
        """Update priorities based on request timers and latency requirements"""
        print("Priority update thread started")
        while self.running:
            try:
                for rnti in list(self.request_timers.keys()):
                    # Initialize priority if not exists
                    if rnti not in self.ue_priorities:
                        self._initialize_ue_priority(rnti)
                        
                    requests = self.request_timers[rnti]
                    if not requests:  # No requests for this UE
                        print(f"No requests for RNTI {rnti}, resetting priority")
                        self.reset_priority(rnti)
                        continue
                    
                    # Get UE's latency requirement
                    latency_req = self.ue_info[rnti]['latency_req']
                    incentive_threshold = latency_req / 2  # First half for incentive mode
                    
                    # Find earliest request
                    earliest_req_id = min(requests.items(), key=lambda x: x[1])[0]
                    timer = requests[earliest_req_id]
                    
                    # Calculate priority based on current state
                    if timer < incentive_threshold:
                        priority = self._calculate_incentive_priority(rnti)
                    elif timer < latency_req:
                        priority = self._calculate_accelerate_priority(rnti)
                    else:
                        priority = 100
                    
                    # Update priority state and scheduler
                    self.ue_priorities[rnti] = priority
                    self.set_priority(rnti, priority)
                    
                    # Debug output
                    if timer % 100 == 0:  # Print every 100ms
                        print(f"RNTI {rnti} - Request {earliest_req_id} - "
                              f"Timer {timer}ms/{latency_req}ms - Priority {priority}")
                    
                time.sleep(0.001)  # 1ms update interval
                
            except Exception as e:
                print(f"Error updating priorities: {e}")

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
        if rnti in self.request_timers and self.request_timers[rnti]:
            # Initialize allocation tracking if needed
            if rnti not in self.request_prb_allocations:
                self.request_prb_allocations[rnti] = {}
            
            # Get earliest request ID
            earliest_req_id = min(self.request_timers[rnti].items(), key=lambda x: x[1])[0]
            
            # Initialize or update PRB allocation for this request
            if earliest_req_id not in self.request_prb_allocations[rnti]:
                self.request_prb_allocations[rnti][earliest_req_id] = 0
            self.request_prb_allocations[rnti][earliest_req_id] += prbs
            
            # Debug output
            print(f"Earliest Request {earliest_req_id} for RNTI {rnti} - "
                  f"Total allocated PRBs: {self.request_prb_allocations[rnti][earliest_req_id]}")

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
