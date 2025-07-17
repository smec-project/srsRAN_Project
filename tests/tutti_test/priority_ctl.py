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
            print(
                f"Connected to scheduler at {self.server_ip}:{self.server_port}"
            )
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def set_priority(self, rnti: str, priority: float):
        if not self.connected:
            print("Not connected to scheduler")
            return False

        # Debug print to see exact RNTI string being sent
        print(f"Raw RNTI string: '{rnti}'")

        # Ensure RNTI is exactly 4 characters with null terminator
        rnti_str = f"{rnti:<4}".encode(
            "ascii"
        )  # Left align, space pad to 4 chars

        # Debug print the bytes being sent
        print(f"Sending bytes: {[hex(x) for x in rnti_str]}")

        # Pack message with explicit format
        msg = struct.pack("=5sdb", rnti_str, priority, False)

        # Debug print the full message
        print(f"Full message bytes: {[hex(x) for x in msg]}")

        try:
            bytes_sent = self.sock.send(msg)
            print(
                f"Set RNTI {rnti} priority to {priority} (sent {bytes_sent} bytes)"
            )
            return True
        except Exception as e:
            print(f"Failed to send priority: {e}")
            return False

    def reset_priority(self, rnti: str):
        if not self.connected:
            print("Not connected to scheduler")
            return False

        # Add padding if RNTI string is shorter than 4 chars
        rnti_str = f"{rnti:0<4}".encode()

        msg = struct.pack("5sdb", rnti_str, 0.0, True)
        try:
            bytes_sent = self.sock.send(msg)
            print(f"Reset RNTI {rnti} priority (sent {bytes_sent} bytes)")
            return True
        except Exception as e:
            print(f"Failed to reset priority: {e}")
            return False

    def close(self):
        if self.connected:
            self.sock.close()
            self.connected = False


def test_scenario(rnti: str):
    controller = PriorityController()

    if not controller.connect():
        return

    try:
        print(f"\nStarting priority increase test for RNTI {rnti}")

        # Test with smaller values first
        test_priorities = [i for i in range(100)]
        for priority in test_priorities:
            print(f"\nTesting priority {priority}")
            controller.set_priority(rnti, priority)
            time.sleep(2)  # Wait longer between tests

        print(f"\nResetting RNTI {rnti} priority")
        controller.reset_priority(rnti)
        time.sleep(2)

    finally:
        controller.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test scheduler priority control"
    )
    parser.add_argument(
        "--ip", default="127.0.0.1", help="Scheduler IP address"
    )
    parser.add_argument("--port", type=int, default=5555, help="Scheduler port")
    parser.add_argument(
        "--rnti", required=True, help="RNTI value in hex (e.g., 47e1)"
    )

    args = parser.parse_args()

    test_scenario(args.rnti)
