"""A series of utility functions for extraction of data from Prometheus."""

from typing import Dict, Any, Tuple
import pandas as pd
from abc import abstractmethod
import numpy as np

from graphnet.utilities.logging import Logger


def compute_visible_inelasticity(mc_truth: pd.DataFrame) -> float:
    """Compute visible inelasticity from event-level truth.

    Visible inelasticity is used as a regression label for reconstructions.

    Args:
        mc_truth: DataFrame containing event-level truth information.

    Returns: visible inelasticity
    """
    # Assumes hadronic energy in GeV, should only be valid for starting tracks
    # Eq. 4.16 in Leif Rädel's thesis, parameters for pi+ from Table D.2
    # Thesis can be found here: https://www.institut3b.physik.rwth-aachen.de/global/show_document.asp?id=aaaaaaaaaapwhjz

    # If not CC, then you get just a shower --> y_vis = 1

    # Check that the fields exists
    required_fields = ["interaction", "final_state_type"]
    for key in required_fields:
        try:
            assert key in mc_truth.keys()
        except AssertionError:
            raise AssertionError(
                "One or more of the required labels"
                f"{required_fields} is not available in the"
                f"event. Got {mc_truth.keys()}"
            )

    final_type_1, final_type_2 = abs(mc_truth["final_state_type"])
    if mc_truth["interaction"] != 1:
        visible_inelasticity = 1.0
    elif not (final_type_1 == 13 or final_type_2 == 13):
        # If not numu CC, the CC component looks like a shower --> y_vis
        visible_inelasticity = 1.0
    else:
        muon_energy = mc_truth["final_state_energy"][
            abs(mc_truth["final_state_type"]) == 13
        ][0]
        hadron_energy = mc_truth["final_state_energy"][
            abs(mc_truth["final_state_type"]) != 13
        ][0]

        em_scale = 1.0 - (1.0 - 0.27273) * (hadron_energy / 0.15581) ** (
            -0.15782
        )
        hadron_energy_em = em_scale * hadron_energy
        visible_inelasticity = hadron_energy_em / (
            hadron_energy_em + muon_energy
        )
    return visible_inelasticity


def get_muon_direction(
    mc_truth: pd.DataFrame, transform_az: bool = False
) -> Tuple[float, float]:
    """Get angles of muon in nu_mu CC events."""
    final_type_1, final_type_2 = abs(mc_truth["final_state_type"])
    if mc_truth["interaction"] != 1:
        muon_zenith = -1
        muon_azimuth = -1
    elif not (final_type_1 == 13 or final_type_2 == 13):
        muon_zenith = -1
        muon_azimuth = -1
    else:
        # CC only
        muon_zenith = mc_truth["final_state_zenith"][
            abs(mc_truth["final_state_type"]) == 13
        ][0]
        muon_azimuth = mc_truth["final_state_azimuth"][
            abs(mc_truth["final_state_type"]) == 13
        ][0]
        if transform_az:
            muon_azimuth += np.pi
    return muon_zenith, muon_azimuth


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


class NaNFilter(PrometheusFilter):
    """Skip events with NaN."""

    def __init__(self, filter_on: str = "photons", column: str = "t") -> None:
        """Skip events with NaN in column.

        Args:
            filter_on: Name of field in file to run filter on. Defaults to
            "photons".
            column: Column to chechk for NaN.
        """
        self._column = column
        super().__init__(filter_on=filter_on)

    def __call__(self, event: Dict[str, Any]) -> bool:
        """Skip events with NaN in column.

        Args:
            event: photons from event.

        Returns:
            False if event should be skipped.
        """
        assert self._column in event.keys()
        if any(np.isnan(event[self._column])):
            return False
        else:
            return True


class MinPhotonCount(PrometheusFilter):
    """Skip events with less than a specific number of photons."""

    def __init__(
        self, filter_on: str = "photons", min_photons: int = 500000
    ) -> None:
        """Skip events with less than a specific number of photons.

        Args:
            filter_on: Name of field in file to run filter on. Defaults to
            "photons".
            min_photons: Maximum number of photons. Defaults to 500000.
        """
        self._min_photons = min_photons
        super().__init__(filter_on=filter_on)

    def __call__(self, event: Dict[str, Any]) -> bool:
        """Skip events with less than a specific number of photons.

        Args:
            event: photons from event.

        Returns:
            False if event should be skipped.
        """
        if len(event[list(event.keys())[0]]) >= self._min_photons:
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
