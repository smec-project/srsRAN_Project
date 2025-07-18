"""Machine learning model inference functionality for PMEC Controller."""

import numpy as np
from typing import Optional, Tuple, Any
from pathlib import Path

from .utils import Logger


class ModelInference:
    """Handles machine learning model loading and inference.
    
    Provides functionality to load trained models and scalers,
    and perform predictions on processed feature data.
    """
    
    def __init__(self, logger: Logger):
        """Initialize the model inference handler.
        
        Args:
            logger: Logger instance for debugging output.
        """
        self.logger = logger
        self.model: Optional[Any] = None
        self.scaler: Optional[Any] = None
        self.is_loaded = False
    
    def load_model(self, model_path: str, scaler_path: str) -> bool:
        """Load the trained model and scaler from file paths.
        
        Args:
            model_path: Path to the trained model file.
            scaler_path: Path to the scaler file.
            
        Returns:
            True if both model and scaler were loaded successfully.
        """
        try:
            import joblib
            
            # Check if files exist
            if not Path(model_path).exists():
                self.logger.log(f"Model file not found: {model_path}")
                return False
            
            if not Path(scaler_path).exists():
                self.logger.log(f"Scaler file not found: {scaler_path}")
                return False
            
            # Load model and scaler
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
            self.is_loaded = True
            self.logger.log(f"Successfully loaded model from {model_path}")
            self.logger.log(f"Successfully loaded scaler from {scaler_path}")
            
            return True
            
        except ImportError as e:
            self.logger.log(f"Failed to import joblib: {e}")
            return False
        except Exception as e:
            self.logger.log(f"Error loading model or scaler: {e}")
            return False
    
    def predict(self, features: np.ndarray) -> Tuple[bool, Optional[float]]:
        """Make a prediction using the loaded model.
        
        Args:
            features: Feature array for prediction.
            
        Returns:
            Tuple of (prediction_success, prediction_value).
            prediction_value is None if prediction fails.
        """
        if not self.is_loaded or self.model is None or self.scaler is None:
            self.logger.log("Model or scaler not loaded, cannot make prediction")
            return False, None
        
        try:
            # Ensure features are in the right shape (1, n_features)
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            
            return True, float(prediction)
            
        except Exception as e:
            self.logger.log(f"Error during model prediction: {e}")
            return False, None
    
    def predict_with_bsr_fallback(
        self, 
        features: np.ndarray,
        current_bsr: float,
        previous_bsr: float
    ) -> Tuple[bool, bool, Optional[float]]:
        """Make prediction with BSR increase fallback logic.
        
        This method combines model prediction with a simple BSR increase check
        as a fallback mechanism for request detection.
        
        Args:
            features: Feature array for model prediction.
            current_bsr: Current BSR value.
            previous_bsr: Previous BSR value.
            
        Returns:
            Tuple of (final_prediction, bsr_increased, model_prediction).
            model_prediction is None if model prediction fails.
        """
        # Check if BSR increased
        bsr_increased = current_bsr > previous_bsr
        
        # Get model prediction
        prediction_success, model_prediction = self.predict(features)
        
        if prediction_success and model_prediction is not None:
            # Final prediction is OR of model prediction and BSR increase
            model_pred_bool = bool(model_prediction)
            final_prediction = model_pred_bool or bsr_increased
            
            self.logger.log(
                f"Model prediction: {model_prediction}, "
                f"BSR increased: {bsr_increased}, "
                f"Final prediction: {final_prediction}"
            )
            
            return final_prediction, bsr_increased, model_prediction
        else:
            # Fallback to BSR increase only
            self.logger.log(
                f"Model prediction failed, using BSR fallback: {bsr_increased}"
            )
            return bsr_increased, bsr_increased, None
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model.
        
        Returns:
            Dictionary with model information.
        """
        info = {
            "is_loaded": self.is_loaded,
            "model_type": None,
            "feature_count": None
        }
        
        if self.is_loaded and self.model is not None:
            try:
                info["model_type"] = type(self.model).__name__
                
                # Try to get feature count from the model
                if hasattr(self.model, "n_features_in_"):
                    info["feature_count"] = self.model.n_features_in_
                elif hasattr(self.model, "feature_importances_"):
                    info["feature_count"] = len(self.model.feature_importances_)
                    
            except Exception as e:
                self.logger.log(f"Error getting model info: {e}")
        
        return info 