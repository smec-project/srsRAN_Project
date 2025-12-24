"""Main entry point for the SMEC Controller application."""

import argparse
import time

from src import SmecController, ControllerConfig


def main():
    """Main function to run the SMEC Controller."""
    parser = argparse.ArgumentParser(description="SMEC Controller")
    parser.add_argument(
        "--log",
        action="store_true",
        default=False,
        help="Enable logging to file (default: False)",
    )
    parser.add_argument(
        "--collect-logs",
        action="store_true",
        default=False,
        help="Only collect logs without priority adjustment (default: False)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Size of the window for analysis (default: 5)",
    )
    parser.add_argument(
        "--slo-ctrl-port",
        type=int,
        default=5557,
        help="Port to receive SLO control plane messages from users (default: 5557)",
    )
    parser.add_argument(
        "--ran-metrics-ip",
        type=str,
        default="127.0.0.1",
        help="IP address for RAN metrics connection (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--ran-metrics-port",
        type=int,
        default=5556,
        help="Port for RAN metrics connection (default: 5556)",
    )
    parser.add_argument(
        "--ran-control-ip",
        type=str,
        default="127.0.0.1",
        help="IP address for RAN control connection (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--ran-control-port",
        type=int,
        default=5555,
        help="Port for RAN control connection (default: 5555)",
    )
    
    args = parser.parse_args()

    # Create configuration from command line arguments
    config = ControllerConfig(
        slo_ctrl_port=args.slo_ctrl_port,
        ran_metrics_ip=args.ran_metrics_ip,
        ran_metrics_port=args.ran_metrics_port,
        ran_control_ip=args.ran_control_ip,
        ran_control_port=args.ran_control_port,
        enable_logging=args.log,
        window_size=args.window_size,
        collect_logs_only=args.collect_logs,
    )

    # Create and start controller
    controller = SmecController(config)

    if controller.start():
        try:
            if args.collect_logs:
                print("SMEC Controller is running in log collection mode (no priority adjustment). Press Ctrl+C to stop.")
            else:
                print("SMEC Controller is running. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down controller...")
        finally:
            controller.stop()
    else:
        print("Failed to start SMEC Controller")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
