import socket
import time
import argparse
import re


class MetricsMonitor:
    def __init__(self, server_ip="127.0.0.1", server_port=5556):
        self.server_ip = server_ip
        self.server_port = server_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected = False

    def connect(self):
        try:
            self.sock.connect((self.server_ip, self.server_port))
            self.connected = True
            print(f"Connected to metrics server at {self.server_ip}:{self.server_port}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def receive_metrics(self):
        if not self.connected:
            print("Not connected to server")
            return None

        try:
            data = self.sock.recv(1024).decode("utf-8")
            if not data:
                return None

            # Parse the metrics string
            metrics = {}
            for line in data.strip().split("\n"):
                values = dict(item.split("=") for item in line.split(","))
                ue_idx = values["UE_IDX"]
                metrics[ue_idx] = {
                    "RNTI": values["RNTI"],
                    "PRBs": values["PRBs"],
                    "SLOT": values["SLOT"],
                }
            return metrics
        except Exception as e:
            print(f"Failed to receive metrics: {e}")
            return None

    def close(self):
        if self.connected:
            self.sock.close()
            self.connected = False


def monitor_metrics():
    monitor = MetricsMonitor()

    if not monitor.connect():
        return

    try:
        print("\nStarting metrics monitoring...")
        print("Format: UE_IDX | RNTI | PRBs | SLOT")
        print("-" * 50)

        while True:
            metrics = monitor.receive_metrics()
            if metrics:
                for ue_idx, values in metrics.items():
                    print(
                        f"{ue_idx:6} | {values['RNTI']:8} | {values['PRBs']:4} | {values['SLOT']}"
                    )

    except KeyboardInterrupt:
        print("\nStopping metrics monitoring...")
    finally:
        monitor.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor UE scheduling metrics")
    parser.add_argument("--ip", default="127.0.0.1", help="Metrics server IP address")
    parser.add_argument("--port", type=int, default=5556, help="Metrics server port")

    args = parser.parse_args()

    monitor_metrics()
