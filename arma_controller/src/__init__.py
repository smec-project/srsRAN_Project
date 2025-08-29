"""Arma Controller Package.

A modular controller system for managing traffic prioritization in 5G RAN
environments with UDP-based communication protocols.
"""

from .controller import ArmaController
from .config import ControllerConfig

__version__ = "1.0.0"
__all__ = ["ArmaController", "ControllerConfig"]