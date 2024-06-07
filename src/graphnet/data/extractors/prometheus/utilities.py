"""A series of utility functions for extraction of data from Prometheus."""

from typing import Dict, Any
import pandas as pd
from abc import abstractmethod

from graphnet.utilities.logging import Logger


class PrometheusFilter(Logger):
    """Generic Filter Class for PrometheusReader."""

    def __init__(self, filter_on: str) -> None:
        """Instantiate filter.

        Args:
            filter_on: Name of field in file to run filter on.
        """
        # Member Variables
        self._filter_on = filter_on
        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

    @abstractmethod
    def __call__(self, event: Dict[str, Any]) -> bool:
        """Return True if the event should be kept.

        Return False i the event should be skipped.
        """


class EmptyFilter(PrometheusFilter):
    """Skip empty events."""

    def __call__(self, event: Dict[str, Any]) -> bool:
        """Skip empty events.

        Args:
            event: photons from event.

        Returns:
            True if event is not empty.
        """
        if len(event[list(event.keys())[0]]) > 1:
            return True
        else:
            return False


class MaxPhotonCount(PrometheusFilter):
    """Skip events exceeding a specific number of photons."""

    def __init__(
        self, filter_on: str = "photons", max_photons: int = 500000
    ) -> None:
        """Skip events exceeding a specific number of photons.

        Args:
            filter_on: Name of field in file to run filter on. Defaults to
            "photons".
            max_photons: Maximum number of photons. Defaults to 500000.
        """
        self._max_photons = max_photons
        super().__init__(filter_on=filter_on)

    def __call__(self, event: Dict[str, Any]) -> bool:
        """Skip events that exceed the maximum number of photons.

        Args:
            event: photons from event.

        Returns:
            False if event exceeds limit on photon count.
        """
        if len(event[list(event.keys())[0]]) < self._max_photons:
            return True
        else:
            return False
