"""SMEC Controller Package.

A modular controller system for managing PETS (Predictive Edge Traffic Scheduling)
in 5G RAN environments.
"""

from .controller import SmecController
from .config import ControllerConfig

__version__ = "1.0.0"
__all__ = ["SmecController", "ControllerConfig"] 