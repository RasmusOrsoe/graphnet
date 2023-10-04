"""Detector-specific modules, for data ingestion and standardisation."""

from .icecube import IceCube86, IceCubeDeepCore
from .prometheus import Prometheus, ORCA150
from .detector import Detector
