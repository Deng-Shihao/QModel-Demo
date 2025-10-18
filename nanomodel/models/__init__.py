"""NanoModel model registry."""
from ._const import get_best_device
from .auto import MODEL_MAP, AutoNanoModel
from .base import BaseNanoModel
from .definitions import *

__all__ = ["BaseQModel", "QuantizationConfig"]
