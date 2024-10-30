"""Modules for reading data files from the Prometheus project."""

from typing import List, Union, OrderedDict, Optional
import pandas as pd
from pathlib import Path
from pyarrow.lib import ArrowInvalid

from graphnet.data.extractors.prometheus import PrometheusExtractor
from graphnet.data.extractors.prometheus.utilities import PrometheusFilter
from .graphnet_file_reader import GraphNeTFileReader


class PrometheusReader(GraphNeTFileReader):
    """A class for reading parquet files from Prometheus simulation."""

    def __init__(
        self, filters: Optional[List[PrometheusFilter]] = None
    ) -> None:
        """Initialize reader.

        Args:
            filters: A list of filters to apply on Prometheus files.
                If a single filter returns False, the event will be skipped.
                Defaults to None.
        """
        # Member Variables
        self._filters = filters
        self._accepted_file_extensions = [".parquet"]
        self._accepted_extractors = [PrometheusExtractor]
        super().__init__()

    def _keep_event(self, extracted_event: OrderedDict) -> bool:
        if self._filters is not None:
            filter_counter = 0
            for filter in self._filters:
                # True + False = 1, True + True = 2
                if filter._filter_on in extracted_event.keys():
                    filter_counter += filter(
                        extracted_event[filter._filter_on]
                    )
                else:
                    self.warning_once(
                        f"{filter._filter_on} not in event. Event skipped."
                    )
                    filter_counter += False
            if filter_counter < len(self._filters):
                # At least 1 filter passed False
                keep_event = False
            else:
                keep_event = True
        else:
            keep_event = True
        return keep_event

    def __call__(self, file_path: str) -> List[OrderedDict]:
        """Extract data from single parquet file.

        Args:
            file_path: Path to parquet file.

        Returns:
            Extracted data.
        """
        # Open file
        outputs = []
        try:
            file = pd.read_parquet(file_path)
            for k in range(len(file)):  # Loop over events in file
                extracted_event = OrderedDict()
                for extractor in self._extractors:
                    assert isinstance(extractor, PrometheusExtractor)
                    if extractor._table in file.columns:
                        try:
                            output = extractor(file[extractor._table][k])
                            extracted_event[extractor._extractor_name] = output
                        except Exception as e:  # noqa
                            self.warning(
                                "Unable to apply "
                                f"{extractor.__class__.__name__} to"
                                f" {file_path}."
                            )

                # Apply filter. If one filter returns False event is skipped.
                if self._keep_event(extracted_event=extracted_event):
                    outputs.append(extracted_event)
        except ArrowInvalid:
            self.error(f"{file_path} appears to be corrupted. Skipping..")
        return outputs

    def find_files(self, path: Union[str, List[str]]) -> List[str]:
        """Search folder(s) for parquet files.

        Args:
            path: directory to search for parquet files.

        Returns:
            List of parquet files in the folders.
        """
        files = []
        if isinstance(path, str):
            path = [path]

        # List of files as Path objects
        for p in path:
            files.extend(
                list(Path(p).rglob(f"*{self.accepted_file_extensions}"))
            )

        # List of files as str's
        paths_as_str: List[str] = []
        for f in files:
            paths_as_str.append(f.absolute().as_posix())

        return paths_as_str
