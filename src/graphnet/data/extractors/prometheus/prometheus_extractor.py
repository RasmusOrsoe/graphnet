"""Parquet Extractor for conversion of simulation files from PROMETHEUS."""
from typing import List, Dict, Any
import pandas as pd
import numpy as np

from graphnet.data.extractors import Extractor
from .utilities import compute_visible_inelasticity, get_muon_direction


class PrometheusExtractor(Extractor):
    """Class for extracting information from PROMETHEUS parquet files.

    Contains functionality required to extract data from PROMETHEUS parquet
    files.
    """

    def __init__(self, extractor_name: str, columns: List[str]):
        """Construct PrometheusExtractor.

        Args:
            extractor_name: Name of the `PrometheusExtractor` instance.
            Used to keep track of the provenance of different data,
            and to name tables to which this data is saved.
            columns: List of column names to extract from the table.
        """
        # Member variable(s)
        self._table = extractor_name
        self._columns = columns
        # Base class constructor
        super().__init__(extractor_name=extractor_name)

    def __call__(self, event: pd.DataFrame) -> Dict[str, Any]:
        """Extract information from parquet file."""
        output = {key: [] for key in self._columns}  # type: ignore
        for key in self._columns:
            if key in event.keys():
                data = event[key]
                if isinstance(data, np.ndarray):
                    data = data.tolist()
                if isinstance(data, list):
                    output[key].extend(data)
                else:
                    output[key].append(data)
            else:
                self.warning_once(f"{key} not found in {self._table}!")
        return output


class PrometheusTruthExtractor(PrometheusExtractor):
    """Class for extracting event level truth from Prometheus parquet files.

    This Extractor will "initial_state" i.e. neutrino truth.
    """

    def __init__(
        self,
        table_name: str = "mc_truth",
        transform_azimuth: bool = True,
    ) -> None:
        """Construct PrometheusTruthExtractor.

        Args:
            table_name: Name of the table in the parquet files that contain
                event-level truth. Defaults to "mc_truth".
            transform_azimuth: Some simulation has the azimuthal angle
            written in a [-pi, pi] projection instead of [0, 2pi].
            If True, the azimuthal angle will be transformed to [0, 2pi].
        """
        columns = [
            "interaction",
            "initial_state_energy",
            "initial_state_type",
            "initial_state_zenith",
            "initial_state_azimuth",
            "initial_state_x",
            "initial_state_y",
            "initial_state_z",
            "bjorken_x",
            "bjorken_y",
        ]
        self._transform_az = transform_azimuth
        super().__init__(extractor_name=table_name, columns=columns)

    def __call__(self, event: pd.DataFrame) -> pd.DataFrame:
        """Extract event-level truth information."""
        # Extract data
        visible_inelasticity = compute_visible_inelasticity(event)
        muon_zenith, muon_azimuth = get_muon_direction(event)
        res = super().__call__(event=event)
        # transform azimuth from [-pi, pi] to [0, 2pi] if wanted
        if self._transform_az:
            if len(res["initial_state_azimuth"]) > 0:
                azimuth = np.asarray(res["initial_state_azimuth"]) + np.pi
                azimuth = azimuth.tolist()  # back to list
                res["initial_state_azimuth"] = azimuth
                muon_azimuth += np.pi
        res["visible_inelasticity"] = [visible_inelasticity]
        res["muon_azimuth"] = [muon_azimuth]
        res["muon_zenith"] = [muon_zenith]
        return res


class PrometheusFeatureExtractor(PrometheusExtractor):
    """Class for extracting pulses/photons from Prometheus parquet files."""

    def __init__(self, table_name: str = "photons"):
        """Construct PrometheusFeatureExtractor.

        Args:
            table_name: Name of table in parquet files that contain the
                photons/pulses. Defaults to "photons".
        """
        columns = [
            "sensor_pos_x",
            "sensor_pos_y",
            "sensor_pos_z",
            "string_id",
            "sensor_id",
            "t",
        ]
        super().__init__(extractor_name=table_name, columns=columns)
