import socket
import struct
import threading
import time
import math
from typing import Dict, Optional
import argparse

class PetsController:
    def __init__(
        self,
        app_port: int = 5557,  # Port to receive application messages
        ran_metrics_ip: str = "127.0.0.1",
        ran_metrics_port: int = 5556,  # Port to receive RAN metrics
        ran_control_ip: str = "127.0.0.1",
        ran_control_port: int = 5555,  # Port to send priority updates
        enable_logging: bool = False,  # Whether to enable logging
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
        self.ue_info: Dict[str, dict] = {}  # RNTI (str) -> {app_id, latency_req, request_size, ue_idx}
        self.request_sequences: Dict[str, list] = {}  # RNTI (str) -> list of sequence numbers
        
        # Change request_timers to store start timestamps
        self.request_start_times: Dict[str, Dict[int, float]] = {}  # RNTI -> {request_id -> start_timestamp}
        
        # Track UE priority states
        self.ue_priorities: Dict[str, float] = {}  # RNTI -> current_priority
        
        # Add mapping between UE_IDX and RNTI
        self.ue_idx_to_rnti = {}  # UE_IDX -> RNTI
        self.rnti_to_ue_idx = {}  # RNTI -> UE_IDX
        
        # Logging setup
        self.enable_logging = enable_logging
        self.log_file = open('controller.txt', 'w') if enable_logging else None

    def start(self):
        """Start the controller and all its connections"""
        self.running = True
        
        # Connect to RAN services
        try:
            self.ran_metrics_socket.connect((self.ran_metrics_ip, self.ran_metrics_port))
            self.ran_control_socket.connect((self.ran_control_ip, self.ran_control_port))
        except Exception as e:
            self.log(f"Failed to connect to RAN services: {e}")
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
                self.log(f"New application connected from {addr}")
                self.app_connections[addr[0]] = conn
                threading.Thread(
                    target=self._handle_app_messages,
                    args=(conn, addr),
                    daemon=True
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
            
            app_id = data.decode('utf-8').strip()
            self.app_connections[app_id] = conn
            self.log(f"Application {app_id} registered from {addr}")
            
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
                        self.log(f"New UE registered - RNTI: {rnti}, UE_IDX: {ue_idx}, "
                              f"Latency Req: {latency_req}ms, Size: {request_size} bytes")

                        self.request_start_times[rnti] = {}
                        
                    elif msg_type == "Start":
                        # Format: "Start|rnti|seq_number"
                        _, rnti, seq_num = msg_parts
                        current_time = time.time()
                        self.log(f"Request {seq_num} from RNTI {rnti} start at {current_time}")

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
                        if rnti in self.request_start_times and seq_num in self.request_start_times[rnti]:
                            start_time = self.request_start_times[rnti][seq_num]
                            elapsed_time_ms = (time.time() - start_time) * 1000  # Convert to ms
                            self.log(f"Request {seq_num} from RNTI {rnti} completed in {elapsed_time_ms:.2f}ms at {time.time()}")
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
                data = self.ran_metrics_socket.recv(1024).decode('utf-8')
                if not data:
                    continue
                
                for line in data.strip().split('\n'):
                    values = dict(item.split('=') for item in line.split(','))
                    msg_type = values['TYPE']
                    
                    if msg_type == 'PRB':
                        self._handle_prb_metrics(values)
                    elif msg_type == 'SR':
                        self._handle_sr_metrics(values)
                    elif msg_type == 'BSR':
                        self._handle_bsr_metrics(values)
                
            except Exception as e:
                self.log(f"Error receiving RAN metrics: {e}")

    def set_priority(self, rnti: str, priority: float):
        """Send priority update to RAN"""
        try:
            # Format RNTI string and pack message in the correct format
            rnti_str = f"{rnti:<4}".encode('ascii')  # Left align, space pad to 4 chars
            msg = struct.pack('=5sdb', rnti_str, priority, False)
            
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
            rnti_str = f"{rnti:<4}".encode('ascii')
            msg = struct.pack('=5sdb', rnti_str, 0.0, True)
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

    def _handle_prb_metrics(self, values):
        """Handle PRB allocation metrics"""
        # Keep RNTI as string
        rnti = values['RNTI'][-4:]  # Just take last 4 chars
        ue_idx = values['UE_IDX']
        slot = int(values['SLOT'])
        prbs = int(values['PRBs'])
        
        # Update UE_IDX <-> RNTI mapping
        self.ue_idx_to_rnti[ue_idx] = rnti
        self.rnti_to_ue_idx[rnti] = ue_idx
        
        # Store basic metrics
        self.current_metrics[rnti] = {
            'UE_IDX': ue_idx,
            'PRBs': prbs,
            'SLOT': slot
        }
        
        self.log(f"PRB received from RNTI=0x{rnti}, slot={slot}, prbs={prbs} at {time.time()}")

    def _handle_sr_metrics(self, values):
        """Handle SR indication metrics"""
        rnti = values['RNTI'][-4:]  # Just take last 4 chars
        slot = int(values['SLOT'])
        self.log(f"SR received from RNTI=0x{rnti}, slot={slot} at {time.time()}")

    def _handle_bsr_metrics(self, values):
        """Handle BSR metrics"""
        rnti = values['RNTI'][-4:]  # Just take last 4 chars
        ue_idx = values['UE_IDX']
        bytes = int(values['BYTES'])
        slot = int(values['SLOT'])
        
        self.log(f"bsr received from RNTI=0x{rnti}, slot={slot}, bytes={bytes} at {time.time()}")

        if rnti not in self.current_metrics:
            self.current_metrics[rnti] = {}
        self.current_metrics[rnti].update({
            'UE_IDX': ue_idx,
            'BSR_BYTES': bytes,
            'BSR_SLOT': slot
        })

    def log(self, message: str):
        """Helper method for logging"""
        if self.enable_logging and self.log_file:
            self.log_file.write(message + "\n")
            self.log_file.flush()

def main():
    parser = argparse.ArgumentParser(description='PETS Controller')
    parser.add_argument('--log', action='store_true', default=False,
                      help='Enable logging to file (default: False)')
    args = parser.parse_args()
    
    controller = PetsController(enable_logging=args.log)
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
