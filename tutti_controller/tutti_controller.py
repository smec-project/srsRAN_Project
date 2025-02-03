import socket
import struct
import threading
import time
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
        threading.Thread(target=self._process_and_update, daemon=True).start()
        
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

                    elif msg_type == "REQUEST":
                        # Format: "REQUEST|RNTI|SEQ_NUM"
                        _, rnti, seq_num = msg_parts
                        seq_num = int(seq_num)
                        if rnti in self.request_sequences:
                            self.request_sequences[rnti].append(seq_num)
                        else:
                            print(f"Warning: Request for unknown RNTI {rnti}")

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
                    metrics[rnti] = {
                        'UE_IDX': values['UE_IDX'],
                        'PRBs': values['PRBs'],
                        'SLOT': values['SLOT']
                    }
                self.current_metrics = metrics
            except Exception as e:
                print(f"Error receiving RAN metrics: {e}")

    def _process_and_update(self):
        """Process metrics and requirements, update priorities"""
        while self.running:
            try:
                # Process each UE's requirements and current state
                for rnti, info in self.ue_info.items():
                    # Get current metrics for this UE using RNTI directly
                    if rnti in self.current_metrics:
                        metrics = self.current_metrics[rnti]
                        
                        # Get pending requests count
                        pending_requests = len(self.request_sequences.get(rnti, []))
                        
                        print(f"UE RNTI {rnti} (IDX {info['ue_idx']}):")
                        print(f"  Latency Requirement: {info['latency_req']}ms")
                        print(f"  Request Size: {info['request_size']} bytes")
                        print(f"  Pending Requests: {pending_requests}")
                        print(f"  Current Metrics: {metrics}")
                        
                        # TODO: Implement priority computation based on:
                        # - Latency requirement
                        # - Request size
                        # - Number of pending requests
                        # - Current metrics
                
                time.sleep(1)
            except Exception as e:
                print(f"Error in processing loop: {e}")

    def set_priority(self, rnti: str, priority: float):
        """Send priority update to RAN"""
        try:
            # Get UE_IDX from current metrics
            if rnti in self.current_metrics:
                ue_idx = int(self.current_metrics[rnti]['UE_IDX'])
                msg = struct.pack('<Hd?', ue_idx, priority, False)
                self.ran_control_socket.send(msg)
                return True
            return False
        except Exception as e:
            print(f"Failed to send priority update: {e}")
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
