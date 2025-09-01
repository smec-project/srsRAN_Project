"""Main SMEC Controller class that orchestrates all components."""

import threading
import time
from typing import Dict, Optional, List, Any

from .config import ControllerConfig, DefaultPaths
from .utils import Logger
from .event_processor import EventProcessor
from .model_inference import ModelInference
from .priority_manager import PriorityManager
from .metrics_processor import MetricsProcessor
from .network_handler import NetworkHandler
from .debug_receiver import DebugReceiver


class SmecController:
    """Main SMEC Controller class.
    
    Orchestrates all components including network handling, event processing,
    model inference, priority management, and metrics processing for
    5G RAN PETS (Predictive Edge Traffic Scheduling).
    """
    
    def __init__(self, config: Optional[ControllerConfig] = None):
        """Initialize the SMEC Controller.
        
        Args:
            config: Configuration settings. Uses default if None.
        """
        self.config = config or ControllerConfig()
        
        # Initialize logger
        self.logger = Logger(
            self.config.enable_logging, 
            DefaultPaths.LOG_FILE
        )
        
        # Initialize components
        self.event_processor = EventProcessor(self.config.window_size, self.logger)
        self.model_inference = ModelInference(self.logger)
        self.priority_manager = PriorityManager(self.config, self.logger)
        
        # Initialize metrics processor with component dependencies
        self.metrics_processor = MetricsProcessor(
            self.event_processor,
            self.priority_manager,
            self.model_inference,
            self.logger
        )
        
        # Initialize network handler
        self.network_handler = NetworkHandler(
            self.config,
            self.priority_manager,
            self.logger
        )
        
        # Set up metrics callback
        self.network_handler.set_metrics_callback(
            self.metrics_processor.process_metrics_data
        )
        
        # Initialize debug receiver
        self.debug_receiver = DebugReceiver(self.logger)
        
        # Controller state
        self.running = False
        self.priority_update_thread: Optional[threading.Thread] = None
        
        self.logger.log("SMEC Controller initialized")
    
    def load_model(self, model_path: Optional[str] = None, scaler_path: Optional[str] = None) -> bool:
        """Load machine learning model and scaler.
        
        Args:
            model_path: Path to the model file. Uses config default if None.
            scaler_path: Path to the scaler file. Uses config default if None.
            
        Returns:
            True if model was loaded successfully.
        """
        model_path = model_path or self.config.model_path or DefaultPaths.MODEL_PATH
        scaler_path = scaler_path or self.config.scaler_path or DefaultPaths.SCALER_PATH
        
        success = self.model_inference.load_model(model_path, scaler_path)
        
        if success:
            model_info = self.model_inference.get_model_info()
            self.logger.log(f"Model loaded: {model_info}")
        
        return success
    
    def start(self) -> bool:
        """Start the SMEC Controller and all its components.
        
        Returns:
            True if controller was started successfully.
        """
        try:
            self.logger.log("Starting SMEC Controller...")
            
            # Load model if paths are configured and not in logs-only mode
            if (self.config.model_path and self.config.scaler_path and 
                not self.config.collect_logs_only):
                self.load_model()
            
            # Start networking
            if not self.network_handler.start_networking():
                self.logger.log("Failed to start network handler")
                return False
            
            # Start debug receiver
            self.debug_receiver.start()
            
            # Set running state
            self.running = True
            
            # Start priority update thread only if not in logs-only mode
            if not self.config.collect_logs_only:
                self.priority_update_thread = threading.Thread(
                    target=self._priority_update_loop,
                    daemon=True
                )
                self.priority_update_thread.start()
                self.logger.log("SMEC Controller started successfully")
            else:
                self.logger.log("SMEC Controller started in log collection mode (no priority updates)")
            
            return True
            
        except Exception as e:
            self.logger.log(f"Error starting SMEC Controller: {e}")
            return False
    
    def _priority_update_loop(self) -> None:
        """Main priority update loop running in a separate thread."""
        self.logger.log("Priority update thread started")
        
        while self.running:
            try:
                # Update all remaining times first
                current_time = time.time()
                self.priority_manager.update_all_remaining_times(current_time)
                
                # Get all registered UEs
                registered_ues = list(self.priority_manager.ue_info.keys())
                
                for rnti in registered_ues:
                    # Get priority update information
                    should_update, new_priority, log_msg = (
                        self.priority_manager.get_priority_update_info(rnti)
                    )
                    
                    if should_update:
                        # Send priority update via network handler
                        if new_priority > 0:
                            success = self.network_handler.set_priority(rnti, new_priority)
                        else:
                            success = self.network_handler.reset_priority(rnti)
                        
                        if success:
                            # Update stored priority
                            self.priority_manager.update_priority(rnti, new_priority)
                            
                            # Log the update
                            # if log_msg:
                            #     self.logger.log(log_msg)
                        else:
                            self.logger.log(f"Failed to send priority update for RNTI {rnti}")
                
            except Exception as e:
                self.logger.log(f"Error in priority update loop: {e}")
            
            # Sleep for the configured interval
            time.sleep(self.config.priority_update_interval)
        
        self.logger.log("Priority update thread stopped")
    
    def stop(self) -> None:
        """Stop the SMEC Controller and clean up all resources."""
        self.logger.log("Stopping SMEC Controller...")
        
        # Set running state to False
        self.running = False
        
        # Stop networking
        self.network_handler.stop_networking()
        
        # Wait for priority update thread to finish (if it exists)
        if (self.priority_update_thread and 
            self.priority_update_thread.is_alive()):
            self.priority_update_thread.join(timeout=1.0)
        
        # Stop debug receiver
        self.debug_receiver.stop()
        
        # Close logger
        self.logger.close()
        
        print("SMEC Controller stopped")
    
    def get_system_status(self) -> Dict[str, any]:
        """Get comprehensive system status information.
        
        Returns:
            Dictionary with system status information.
        """
        status = {
            "controller_running": self.running,
            "config": {
                "window_size": self.config.window_size,
                "enable_logging": self.config.enable_logging,
                "priority_update_interval": self.config.priority_update_interval,
                "collect_logs_only": self.config.collect_logs_only,
            },
            "network": self.network_handler.get_connection_status(),
            "model": self.model_inference.get_model_info(),
            "registered_ues": len(self.priority_manager.ue_info),
            "active_priorities": len([
                rnti for rnti, priority in self.priority_manager.ue_priorities.items()
                if priority > 0
            ])
        }
        
        return status
    
    def get_ue_status(self, rnti: int) -> Optional[Dict[str, Any]]:
        """Get detailed status for a specific UE.
        
        Args:
            rnti: Radio Network Temporary Identifier as integer.
            
        Returns:
            Dictionary with UE status or None if UE not found.
        """
        if rnti not in self.priority_manager.ue_info:
            return None
        
        status = {
            "rnti": rnti,
            "ue_info": self.priority_manager.ue_info[rnti],
            "current_priority": self.priority_manager.ue_priorities.get(rnti, 0.0),
            "latest_bsr": self.priority_manager.ue_latest_bsr.get(rnti, 0),
            "active_requests": len(self.priority_manager.ue_remaining_times.get(rnti, [])),
            "total_predictions": self.priority_manager.positive_predictions.get(rnti, 0),
            "remaining_times_summary": self.priority_manager.get_remaining_times_summary(rnti),
            "metrics_summary": self.metrics_processor.get_ue_metrics_summary(rnti)
        }
        
        return status
    
    def list_registered_ues(self) -> List[int]:
        """Get list of all registered UE RNTIs.
        
        Returns:
            List of RNTI integers.
        """
        return list(self.priority_manager.ue_info.keys())
    
    def print_window_data(self, rnti: int) -> None:
        """Print window data for debugging purposes.
        
        Args:
            rnti: Radio Network Temporary Identifier as integer.
        """
        self.event_processor.print_window_data(rnti)
    
    def force_priority_update(self, rnti: int, priority: float) -> bool:
        """Force a priority update for a specific UE.
        
        Args:
            rnti: Radio Network Temporary Identifier as integer.
            priority: Priority value to set.
            
        Returns:
            True if update was successful.
        """
        if rnti not in self.priority_manager.ue_info:
            self.logger.log(f"Cannot update priority for unknown RNTI 0x{rnti:x}")
            return False
        
        if priority > 0:
            success = self.network_handler.set_priority(rnti, priority)
        else:
            success = self.network_handler.reset_priority(rnti)
        
        if success:
            self.priority_manager.update_priority(rnti, priority)
            self.logger.log(f"Forced priority update for RNTI 0x{rnti:x}: {priority}")
        
        return success
    
    def cleanup_ue(self, rnti: int) -> bool:
        """Clean up all data for a specific UE.
        
        Args:
            rnti: Radio Network Temporary Identifier as integer.
            
        Returns:
            True if cleanup was successful.
        """
        if rnti not in self.priority_manager.ue_info:
            return False
        
        # Reset priority first
        self.network_handler.reset_priority(rnti)
        
        # Clean up in priority manager
        self.priority_manager.cleanup_ue(rnti)
        
        # Clean up in event processor
        if rnti in self.event_processor.window_events:
            del self.event_processor.window_events[rnti]
        if rnti in self.event_processor.ue_base_times:
            del self.event_processor.ue_base_times[rnti]
        
        # Clean up in metrics processor
        if rnti in self.metrics_processor.ue_bsr_events:
            del self.metrics_processor.ue_bsr_events[rnti]
        if rnti in self.metrics_processor.ue_last_bsr:
            del self.metrics_processor.ue_last_bsr[rnti]
        if rnti in self.metrics_processor.ue_peak_buffer_size:
            del self.metrics_processor.ue_peak_buffer_size[rnti]
        
        self.logger.log(f"Cleaned up all data for RNTI 0x{rnti:x}")
        return True 