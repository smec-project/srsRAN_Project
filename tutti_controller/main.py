"""Main entry point for the Tutti Controller application."""

import argparse
import time

from src import TuttiController, ControllerConfig


def main():
    """Main function to run the Tutti Controller."""
    parser = argparse.ArgumentParser(description="Tutti Controller")
    parser.add_argument(
        "--app-port",
        type=int,
        default=5557,
        help="Port to receive application messages (default: 5557)",
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
    parser.add_argument(
        "--log",
        action="store_true",
        help="Enable logging to controller.log",
    )
    
    args = parser.parse_args()

    # Create configuration from command line arguments
    config = ControllerConfig(
        app_port=args.app_port,
        ran_metrics_ip=args.ran_metrics_ip,
        ran_metrics_port=args.ran_metrics_port,
        ran_control_ip=args.ran_control_ip,
        ran_control_port=args.ran_control_port,
        log_file_path="controller.log" if args.log else None,
    )

    # Create and start controller
    controller = TuttiController(config)

    if controller.start():
        try:
            print("Tutti Controller is running. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down controller...")
        finally:
            controller.stop()
    else:
        print("Failed to start Tutti Controller")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())