"""Turn Photons from Prometheus Simulation into Pulsemaps."""

from typing import List, Dict, Union, Tuple
import pandas as pd
import numpy as np

from .prometheus_extractor import PrometheusExtractor


class PulsemapSimulator(PrometheusExtractor):
    """Turn raw photons from PrometheusSimulation into observable pulses.

    PulsemapSimulator applies a simplistic simulation of DOM response to the
    Prometheus simulation. The simulation includes:

        1. A simple trigger window
        2. Optical efficiency downsampling
        3. Merging of coincident photons to mimmick transit time spread.
        4. Adding an idealistic charge variable
        5. Smearing both arrival time and charge of resulting pulses.

    This method is ideal for use cases where the training/testing is supposed
    to mimmick real-life detector simulation, but contains significant
    simplifications.

    This method does not include:
        a) Angular acceptance curves
        b) DAQ-simulation and dark noise
        c) PMT-response
    """

    def __init__(
        self,
        pulsemap_name: str,
        noise_rate: float,
        optical_efficiency: float,
        geometry_table: pd.DataFrame,
        merge_window: float = 1.0,
        charge_smearing_std: float = 1.0,
        time_smearing_std: float = 1.0,
        minimum_trigger_window: Union[float, int] = 1000,
        x_column: str = "sensor_pos_x",
        y_column: str = "sensor_pos_y",
        z_column: str = "sensor_pos_z",
        time_column: str = "t",
        string_id_column: str = "string_id",
        seed: int = 42,
        photon_table: str = "photons",
        remove_events_larger_than: int = 100000,
        min_photons: int = 4,
    ) -> None:
        """Simulate a pulsemap from raw simulated photons.

        The PulsemapSimulator will construct a semi-realistic pulsemap
        from simulated photons. Specifically, this method will:

        1. Apply module efficiencies to the signal photons.
            (I.e. a large portion of observed photons are randomly dropped)
        2. Add noise photons as per given noise rate.
        3. Merge photons (both noise and signal) that fall within the merge
            window (by default 1ns).
        4. Add a pseudo charge (Observed photons + smearing)
        5. Smear merged arrival times of pulses to mimmick arrival time spread.

        The output represents a single data-stream from an optical module,
        interspersed with noise and smearing. I.e. per-pmt information is not
        available.

        This method does not include:
        a) Angular acceptance curves
        b) DAQ-simulation and dark noise
        c) PMT-response

        Args:
            pulsemap_name: Name of the pulsemap table.
            noise_rate: Total noise rate of a single optical module in Hz.
            optical_efficiency: Overall efficency of optical module. Expected
            to be in range ]0, 1]. This value is used to mimmick optical
            module efficiency. If 1 all photons are kept.
            geometry_table: A pd.DataFrame containing the xyz positions.
            merge_window: The time window in which coincident photons will be
            merged on the same optical module. I.e. photons that arrive within
            this window on the same module will be merged into a single pulse.
            (Defaults to 1ns).
            charge_smearing_std: The std of the Gaussian used to
            smear the pseudo-charge. If 0, no smearing is done. Defaults to
            1 p.e.
            time_smearing_std: The std of the Gaussian used to smear the
            arrival time of the pulse. If 0, no smearing is done. Defaults to
            1 ns.
            minimum_trigger_window: Minimal trigger window used to calculate
            x_column: Name of x-coordinate of module position in files.
            defaults to "sensor_pos_z".
            y_column: Name of y-coordinate of module position in files.
            defaults to "sensor_pos_y"
            z_column: name of z-coordinate of module positions in files.
            defaults to "sensor_pos_z"
            time_column: name of arrival time column in files.
            defaults to "t".
            noise contributions. Defaults to 1000ns.
            seed: seed used for rng. Defaults to 42.
            photon_table: Name of the field in files containing the photons.
            remove_events_larger_than: Events containing more photons than this
            will be automatically removed.
            string_id_column: Name of string column in files. Defaults to
            "string_id".
            min_photons: Minimal number of photons required to process event.
            Events with less than `min_photons` will not be processed.
        """
        # Build colum name list for FeatureExtractor
        columns = [x_column, y_column, z_column, time_column, string_id_column]

        # Checks
        assert (optical_efficiency > 0) & (optical_efficiency <= 1.0)

        # Member Variables
        self._noise_rate = noise_rate
        self._efficiency = optical_efficiency
        self._geometry_table = geometry_table
        self._merge_window = merge_window
        self._charge_std = charge_smearing_std
        self._time_std = time_smearing_std
        self._generator = np.random.default_rng(seed=seed)
        self._trigger_window = minimum_trigger_window
        self._x = x_column
        self._y = y_column
        self._z = z_column
        self._string_col = string_id_column
        self._time_column = time_column
        self._max_photons = remove_events_larger_than
        self._min_photons = min_photons
        super().__init__(extractor_name=photon_table, columns=columns)

        # Overwrite fields set by super().
        self._table = photon_table
        self._extractor_name = pulsemap_name

    def __call__(self, event: pd.DataFrame) -> pd.DataFrame:
        """Process photons from single event into pulses."""
        # Extract raw photons from event
        photons = super().__call__(event=event)

        # Create empty variables - these will be returned if needed
        features = self._columns + ["charge", "is_signal"]
        pulses: Dict[str, List] = {feature: [] for feature in features}

        # Return empty if not enough signal
        if (self._get_length(photons) >= self._min_photons) & (
            self._get_length(photons) <= self._max_photons
        ):

            # Return empty if contains NaN
            if sum(np.isnan(photons[self._time_column])) == 0:

                # Only apply OM efficiency if less than 1.0 (to save time)
                if self._efficiency < 1.0:
                    photons = self._apply_efficiency(photons=photons)

                # Return empty if not enough signal after efficiency
                if self._get_length(photons) > 1:

                    # Apply Trigger window
                    # Centers the trigger window around the signal
                    photons, high = self._apply_trigger_window(photons=photons)

                    # Add noise photons
                    if self._noise_rate > 0:
                        photons = self._add_noise(photons=photons, high=high)
                    else:
                        photons["is_signal"] = np.repeat(
                            1, len(photons[self._time_column])
                        ).tolist()

                    photons = self._sort_by(
                        photons=photons, column=self._time_column
                    )

                    # Smear arrival time
                    if self._time_std > 0:
                        photons[self._time_column] = abs(
                            self._smear_feature(
                                x=photons[self._time_column],
                                std=self._time_std,
                            )
                        ).tolist()

                    # Merge photons into pulses, add charge & delete photons
                    pulses = self._merge_into_pulses(photons=photons)
                    del photons  # save memory

                    # Smear Charge
                    if self._charge_std > 0:
                        pulses["charge"] = abs(
                            self._smear_feature(
                                x=pulses["charge"], std=self._charge_std
                            )
                        )
                    return pulses
                else:
                    return self._make_empty_return()
            else:
                return self._make_empty_return()
        else:
            return self._make_empty_return()

    def _make_empty_return(self) -> Dict[str, List]:
        features = self._columns + ["charge", "is_signal"]
        pulses: Dict[str, List] = {feature: [] for feature in features}
        return pulses

    def _smear_feature(self, x: List[float], std: float) -> np.ndarray:
        """Smear a list of features using a Gaussian distribution."""
        return self._generator.normal(loc=x, scale=std, size=len(x))

    def _merge_into_pulses(self, photons: Dict[str, List]) -> Dict[str, List]:
        """Merge photon attributes into pulses and add pseudo-charge."""
        # Create temporary module ids based on xyz coordinates
        ids = self._assign_temp_ids(
            x=photons["sensor_pos_x"],
            y=photons["sensor_pos_y"],
            z=photons["sensor_pos_z"],
        )

        # Identify photons that needs to be merged
        idx = self._find_photons_for_merging(
            t=photons["t"], ids=ids, merge_window=self._merge_window
        )

        # Merge photon attributes based on temporary ids
        pulses = self._merge_to_pulses(data_dict=photons, ids_to_merge=idx)

        # Assign pseudo-charge based on number of observed p.e.
        charge = [len(elem) for elem in idx]  # May include noise
        pulses["charge"] = charge

        # Delete photons that was merged
        delete_these = []
        for group in idx:
            delete_these.extend(group)

        if len(delete_these) > 0:
            for key in photons.keys():
                photons[key] = np.delete(
                    np.array(photons[key]), delete_these
                ).tolist()
        photons["charge"] = np.repeat(
            1, len(photons[list(photons.keys())[0]])
        ).tolist()

        # Add the pulses instead
        for key in photons.keys():
            photons[key].extend(pulses[key])
        del pulses  # save memory

        return photons

    def _add_noise(
        self, photons: Dict[str, List], high: float
    ) -> Dict[str, List]:
        """Sample stochastic noise and add to observed photons."""
        # Dict for handling noise photons
        noise_photons = {}

        # Identify time window
        times = photons[self._time_column]

        # Convert rate & time window to expected noise count
        module_expectation = self._noise_rate * (
            self._trigger_window / 1e9
        )  # ns -> seconds
        total_expectation = module_expectation * len(self._geometry_table)

        # Sample number of noise photons
        n_noise = self._generator.poisson(lam=total_expectation, size=1)

        # Sample uniform distribution for arrival times of noise photons
        noise_times = self._generator.uniform(
            low=0, high=high, size=n_noise
        ).tolist()

        noise_photons[self._time_column] = noise_times

        # Sample random sample for module position of noise photons
        module_ids = np.arange(0, len(self._geometry_table), 1)
        idx = self._generator.choice(
            a=module_ids,
            replace=True,  # May happen on same OM
            shuffle=False,
            size=n_noise,
        )
        for key in [self._x, self._y, self._z, self._string_col]:
            coordinate = self._geometry_table.loc[idx, key].tolist()
            noise_photons[key] = coordinate

        # Handle signal/noise labels 1 = Signal, 0 = Noise
        photons["is_signal"] = np.repeat(1.0, len(times)).tolist()
        noise_photons["is_signal"] = np.repeat(0.0, n_noise).tolist()

        # Merge dictionaries
        for key in noise_photons.keys():
            assert key in photons.keys()  # else something is very wrong
            photons[key].extend(noise_photons[key])

        return photons

    def _sort_by(
        self, photons: Dict[str, List], column: str
    ) -> Dict[str, List]:
        """Sort Dictionary by field."""
        # Grab Column
        column_array = np.array(photons[column])

        # Sort values
        idx = column_array.argsort()
        for key in photons.keys():
            photons[key] = np.array(photons[key])[idx].tolist()

        return photons

    def _apply_trigger_window(
        self, photons: Dict[str, List]
    ) -> Tuple[Dict[str, List], float]:
        # Identify time window
        times = photons[self._time_column]
        event_window = np.max(times) - np.min(times)

        # Choose event window if larger than trigger window
        if event_window >= self._trigger_window:
            time_window = event_window
        else:
            # choose minimal trigger window if smaller
            time_window = self._trigger_window

        mean_signal_time = np.mean(times)
        low = mean_signal_time - time_window / 2
        high = mean_signal_time + time_window / 2

        # Adjust arrival time
        times_array = np.array(times)
        del times  # MyPy wanted this solution..
        times_array = times_array + np.abs(low)  # Center to trigger window
        photons[self._time_column] = times_array.tolist()
        return photons, high

    def _apply_efficiency(self, photons: Dict[str, List]) -> Dict[str, List]:
        """Apply module efficiency by dropping photons.

        Method does not offer upsampling, therefore requires an efficiency
        lower than 1.0.
        """
        # Get number of signal photons
        n_photons = self._get_length(photons)

        # Calculate observed number of photons
        scores = self._generator.uniform(low=0, high=1.0, size=n_photons)

        # Subsample features
        sub_sampled_idx = scores <= self._efficiency

        # In-place overwrite of features to save memory
        for key in photons.keys():
            photons[key] = np.array(photons[key])[sub_sampled_idx].tolist()

        return photons

    def _merge_to_pulses(
        self, data_dict: Dict[str, List], ids_to_merge: List[List[int]]
    ) -> Dict[str, List]:
        """Merge photon attributes into pulses according to assigned ids."""
        # Initialize a new dictionary to store the merged results
        merged_dict: Dict[str, List] = {key: [] for key in data_dict.keys()}

        # Iterate over the groups of IDs to merge
        for group in ids_to_merge:
            for key in data_dict.keys():
                # Extract the values corresponding to the current group of IDs
                values_to_merge = [data_dict[key][i] for i in group]

                # Handle numeric and non-numeric fields differently
                if all(
                    isinstance(value, (int, float))
                    for value in values_to_merge
                ):
                    # For numeric fields, calculate the mean
                    merged_value = sum(values_to_merge) / len(values_to_merge)
                else:
                    # For non-numeric fields, join the values into a single string
                    merged_value = " ".join(map(str, values_to_merge))

                merged_dict[key].append(merged_value)

        return merged_dict

    def _assign_temp_ids(
        self, x: List[float], y: List[float], z: List[float]
    ) -> List[int]:
        """Create a temporary module id based on xyz positions."""
        # Convert lists to a structured NumPy array
        data = np.array(
            list(zip(x, y, z)),
            dtype=[("x", float), ("y", float), ("z", float)],
        )

        # Get the unique rows and the indices to reconstruct
        # the original array with IDs
        _, ids = np.unique(data, return_inverse=True, axis=0)

        return ids.tolist()

    def _find_photons_for_merging(
        self, t: List[float], ids: List[int], merge_window: float
    ) -> List[List[int]]:
        """Identify photons that needs to be merged."""
        # Convert lists to a structured NumPy array
        data = np.array(
            list(zip(t, ids)), dtype=[("time", float), ("id", int)]
        )

        # Get original indices after sorting by ID first and then by time
        sorted_indices = np.argsort(data, order=["id", "time"])
        sorted_data = data[sorted_indices]

        close_elements_indices = []
        current_group = [sorted_indices[0]]

        for i in range(1, len(sorted_data)):
            current_value = sorted_data[i]["time"]
            current_id_value = sorted_data[i]["id"]

            # Compare with the last element in the current group
            if (
                current_id_value == sorted_data[i - 1]["id"]
                and current_value - sorted_data[i - 1]["time"] < merge_window
            ):
                current_group.append(sorted_indices[i])
            else:
                # If the group has more than one element, add it to the results
                if len(current_group) > 1:
                    close_elements_indices.append(current_group)
                # Start a new group
                current_group = [sorted_indices[i]]

        # Append the last group if it has more than one element
        if len(current_group) > 1:
            close_elements_indices.append(current_group)

        return close_elements_indices

    def _get_length(self, photons: Dict[str, List]) -> int:
        """Get the length of the event.

        Requires all fields in the Dictionary to have equal lengths.
        """
        lengths = []
        for key in photons.keys():
            lengths.append(len(photons[key]))
        lengths_array = np.array(lengths)
        # Check that lengths are equal
        try:
            assert sum(lengths_array - lengths_array[0]) == 0
        except AssertionError as e:
            print(lengths_array, flush=True)
            raise e
        return lengths_array.tolist()[0]
