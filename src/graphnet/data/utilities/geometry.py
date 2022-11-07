import pandas as pd
import numpy as np


def convert_f2k_to_geometry_table(custom_f2k: str, source: str = "prometheus"):
    """Converts f2k files from simulation engines to graphnet compliant geometry tables. Other, future sources must comply with naming convention of the xyz of the sensors.

    Args:
        custom_f2k (str): path to f2k file
        outdir (str): the directory to which the file is saved_any
        source (str, optional): the source of the f2k file. Defaults to 'prometheus'.
    """
    if source == "prometheus":
        geometry_table = pd.read_csv(
            custom_f2k, sep="\t", lineterminator="\n", header=None
        )
        geometry_table.columns = [
            "hash_1",
            "hash_1",
            "sensor_x",
            "sensor_y",
            "sensor_z",
            "string_idx",
            "pmt_idx",
        ]

    if source == "prometheus_v2":
        geometry_table = pd.read_csv(
            custom_f2k, sep="\t", skiprows=(0, 1, 2, 3), header=None
        )
        geometry_table.columns = [
            "dom_x",
            "dom_y",
            "dom_z",
            "string_idx",
            "pmt_idx_string",
        ]
        geometry_table["pmt_idx_global"] = np.arange(0, len(geometry_table), 1)

    return geometry_table
