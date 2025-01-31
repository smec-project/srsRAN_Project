import socket
import struct
import time
import threading
import argparse

class PriorityController:
    def __init__(self, server_ip="192.168.2.2", server_port=5555):
        self.server_ip = server_ip
        self.server_port = server_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected = False
        
    def connect(self):
        try:
            self.sock.connect((self.server_ip, self.server_port))
            self.connected = True
            print(f"Connected to scheduler at {self.server_ip}:{self.server_port}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
            
    def set_priority(self, ue_index, priority):
        if not self.connected:
            print("Not connected to scheduler")
            return False
            
        msg = struct.pack('<Hd?', ue_index, priority, False)  # H:uint16, d:double, ?:bool
        try:
            bytes_sent = self.sock.send(msg)
            print(f"Set UE{ue_index} priority to {priority} (sent {bytes_sent} bytes)")
            return True
        except Exception as e:
            print(f"Failed to send priority: {e}")
            return False
            
    def reset_priority(self, ue_index):
        if not self.connected:
            print("Not connected to scheduler")
            return False
            
        msg = struct.pack('<Hd?', ue_index, 0.0, True)
        try:
            bytes_sent = self.sock.send(msg)
            print(f"Reset UE{ue_index} priority (sent {bytes_sent} bytes)")
            return True
        except Exception as e:
            print(f"Failed to reset priority: {e}")
            return False
            
    def reset_priority(self, ue_index):
        if not self.connected:
            print("Not connected to scheduler")
            return False
            
        msg = struct.pack('=Hdb', ue_index, 0.0, True)
        try:
            self.sock.send(msg)
            print(f"Reset UE{ue_index} priority")
            return True
        except Exception as e:
            print(f"Failed to reset priority: {e}")
            return False
            
    def close(self):
        if self.connected:
            self.sock.close()
            self.connected = False

def test_scenario():
    controller = PriorityController()
    
    if not controller.connect():
        return
        
    try:
        print("\nStarting priority increase test for UE1")
        # Gradually increase priority for UE1
        for i in range(100):
            priority = 1.0 + i * 1.0  # Start from 1.0 and increase by 1.0 each time
            controller.set_priority(0, priority)
            time.sleep(1)  # Wait 1 second between updates
            
        print("\nResetting UE1 priority")
        controller.reset_priority(1)
        time.sleep(5)  # Wait to observe the reset effect
            
    finally:
        controller.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test scheduler priority control')
    parser.add_argument('--ip', default='127.0.0.1', help='Scheduler IP address')
    parser.add_argument('--port', type=int, default=5555, help='Scheduler port')
    
    args = parser.parse_args()
    
    test_scenario()