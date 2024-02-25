"""Module containing experiment-specific dataclasses."""

from typing import List
from dataclasses import dataclass

from graphnet.deployment.i3modules import I3InferenceModule


@dataclass
class I3FileSet:  # noqa: D101
    i3_file: str
    gcd_file: str


@dataclass
class Settings:
    """Dataclass for workers in I3Deployer."""

    i3_files: List[str]
    gcd_file: str
    output_folder: str
    modules: List[I3InferenceModule]