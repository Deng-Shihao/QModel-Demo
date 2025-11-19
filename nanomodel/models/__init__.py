"""NanoModel model registry."""

from ._const import get_best_device
from .auto import MODEL_DICT, AutoNanoModel
from .base import BaseNanoModel
from .definitions import *
