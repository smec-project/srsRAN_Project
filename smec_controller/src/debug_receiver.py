import socket
import struct
import threading
import time
from .utils import Logger

class DebugReceiver:
    """UDP server to receive debug info (int request index) and log timestamped messages."""
    def __init__(self, logger: Logger, listen_port: int = 6000):
        self.logger = logger
        self.listen_port = listen_port
        self.running = False
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)

    def _run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", self.listen_port))
        sock.settimeout(1.0)
        while self.running:
            try:
                data, addr = sock.recvfrom(8)  # Expecting 8 bytes (2x uint32_t)
                if len(data) == 8:
                    request_index, socket_index = struct.unpack('<II', data)
                    timestamp = time.time()
                    self.logger.log(f"ue{socket_index} sent request {request_index} at {timestamp:.6f}")
            except socket.timeout:
                continue
            except Exception as e:
                self.logger.log(f"DebugReceiver error: {e}")
        sock.close() 