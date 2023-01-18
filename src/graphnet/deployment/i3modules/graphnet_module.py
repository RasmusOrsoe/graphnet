"""Class(es) for deploying GraphNeT models in icetray as I3Modules."""

import os.path
from typing import TYPE_CHECKING, Any, List, Union

import numpy as np
import torch
import dill
from torch_geometric.data import Data

from graphnet.data.extractors import (
    I3FeatureExtractor,
    I3FeatureExtractorIceCube86,
    I3FeatureExtractorIceCubeDeepCore,
    I3FeatureExtractorIceCubeUpgrade,
)
from graphnet.data.constants import FEATURES
from graphnet.models import StandardModel
from graphnet.models.model import Model
from graphnet.utilities.imports import has_icecube_package


if has_icecube_package() or TYPE_CHECKING:
    from icecube.dataclasses import I3Particle, I3Constants
    from I3Tray import *
    from icecube import dataclasses, dataio
    from icecube.icetray import (
        I3Module,
        I3Frame,
    )  # pyright: reportMissingImports=false
    from icecube.dataclasses import (
        I3Double,
    )  # pyright: reportMissingImports=false


class GraphNeTModuleBase(I3Module):
    """Base I3Module for running graphnet models in I3Tray chains."""

    # Class variables
    FEATURES: List[str]
    I3FEATUREEXTRACTOR_CLASS: type
    DTYPES = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }

    def __init__(self, context: Any) -> None:
        """Construct `GraphNeTModuleBase`."""
        # Check
        if self.FEATURES is None:
            raise Exception("Please use an experiment-specific I3Module.")

        # Base class constructor
        I3Module.__init__(self, context)

        # Parameters to `I3Tray.Add(..., param=...)`
        self.AddParameter("keys", "doc_string__key", None)
        self.AddParameter("gcd_file", "doc_string__gcd_file", None)
        self.AddParameter("model", "doc_string__model", None)
        self.AddParameter("pulsemaps", "doc_string__pulsemaps", None)
        self.AddParameter("dtype", "doc_string__dtype", "float32")

        # Standard member variables
        self.keys: Union[str, List[str]]
        self.model: Model
        self.dtype: torch.dtype

    def Configure(self) -> None:  # pylint: disable=invalid-name
        """Configure I3Module based on keyword parameters."""
        # Extract parameters
        keys: Union[str, List[str]] = self.GetParameter("keys")
        gcd_file: str = self.GetParameter("gcd_file")
        model: Union[str, Model] = self.GetParameter("model")
        pulsemaps: Union[str, List[str]] = self.GetParameter("pulsemaps")
        dtype: str = self.GetParameter("dtype")

        # Check(s)
        assert keys is not None
        assert model is not None
        assert gcd_file is not None
        assert pulsemaps is not None
        assert dtype in self.DTYPES
        if isinstance(model, str):
            assert os.path.exists(model)
        assert isinstance(keys, (str, list, tuple))

        if isinstance(pulsemaps, str):
            pulsemaps = [pulsemaps]

        # Set member variables
        self.keys = keys
        self.dtype = self.DTYPES[dtype]

        self.i3extractors = [
            self.I3FEATUREEXTRACTOR_CLASS(pulsemap) for pulsemap in pulsemaps
        ]
        for i3extractor in self.i3extractors:
            i3extractor.set_files(None, gcd_file)

        if isinstance(model, str):
            self.model = StandardModel.load(model)
        else:
            self.model = model

        # Toggle inference mode on, to ensure that any transforms of the model
        # predictions are applied.
        self.model.inference()

    def Physics(
        self, frame: I3Frame
    ) -> None:  # py-l-i-n-t-:- -d-i-s-able=invalid-name
        """Process Physics I3Frame and write predictions."""
        # Extract features
        features = self._extract_feature_array_from_frame(frame)

        # Prepare graph data
        n_pulses = torch.tensor([features.shape[0]], dtype=torch.int32)
        data = Data(
            x=torch.tensor(features, dtype=self.dtype),
            edge_index=None,
            batch=torch.zeros(
                features.shape[0], dtype=torch.int64
            ),  # @TODO: Necessary?
            features=self.FEATURES,
        )

        # @TODO: This sort of hard-coding is not ideal; all features should be
        #        captured by `FEATURES` and included in the output of
        #        `I3FeatureExtractor`.
        data.n_pulses = n_pulses

        # Perform inference
        try:
            predictions = [p.detach().numpy()[0, :] for p in self.model(data)]
            predictions = np.concatenate(
                predictions
            )  # @TODO: Special case for single task
        except:  # noqa: E722
            print("data:", data)
            raise

        # Write predictions to frame
        frame = self._write_predictions_to_frame(frame, predictions)
        self.PushFrame(frame)

    def _extract_feature_array_from_frame(self, frame: I3Frame) -> np.array:
        features = None
        for i3extractor in self.i3extractors:
            feature_dict = i3extractor(frame)
            features_pulsemap = np.array(
                [feature_dict[key] for key in self.FEATURES]
            ).T
            if features is None:
                features = features_pulsemap
            else:
                features = np.concatenate(
                    (features, features_pulsemap), axis=0
                )
        return features

    def _write_predictions_to_frame(
        self, frame: I3Frame, prediction: np.array
    ) -> I3Frame:
        nb_preds = prediction.shape[0]
        if isinstance(self.keys, str):
            if nb_preds > 1:
                keys = [f"{self.keys}_{ix}" for ix in range(nb_preds)]
            else:
                keys = [self.keys]
        else:
            assert (
                len(self.keys) == nb_preds
            ), f"Number of key-names ({len(keys)}) doesn't match number of predictions ({nb_preds})"
            keys = self.keys

        for ix, key in enumerate(keys):
            frame[key] = I3Double(np.float64(prediction[ix]))
        return frame


class PulseCleanerModule:
    """Will Clean your pulses."""

    def __init__(
        self,
        model_path: str,
        pulsemap: str,
        features: List[str],
        pulsemap_extractor: Union[
            List[I3FeatureExtractor], I3FeatureExtractor
        ],
        model_name: str,
        threshold: float = 0.7,
    ):
        """Will Clean your pulses."""
        self.model = torch.load(
            model_path, pickle_module=dill, map_localization="cpu"
        )
        self._pulsemap = pulsemap
        self._features = features
        self._predictions_key = f"{pulsemap}_{model_name}_Predictions"
        self._total_pulsemap_name = f"{pulsemap}_{model_name}_Pulses"
        self._threshold = threshold
        if isinstance(pulsemap_extractor, list):
            self._i3_extractor = pulsemap_extractor
        else:
            self._i3_extractor = [pulsemap_extractor]

    def __call__(self, frame: I3Frame, gcd_file: Any) -> bool:
        """Add a cleaned pulsemap to frame."""
        # inference
        graph = self._make_graph(frame)
        predictions = self._inference(graph)

        # submission methods
        frame = self._submit_predictions(frame, predictions)
        frame = self._add_meta_data(frame)
        frame = self._add_pulsemap_for_each_dom_type(frame, gcd_file)

        return True

    def _inference(self, graph: Data) -> np.ndarray:
        # Perform inference
        return self.model(graph).detach().numpy()

    def _make_graph(
        self, frame: I3Frame
    ) -> Data:  # py-l-i-n-t-:- -d-i-s-able=invalid-name
        """Process Physics I3Frame into graph."""
        # Extract features
        features = self._extract_feature_array_from_frame(frame)

        # Prepare graph data
        n_pulses = torch.tensor([features.shape[0]], dtype=torch.int32)
        data = Data(
            x=torch.tensor(features, dtype=torch.float32),
            edge_index=None,
            batch=torch.zeros(
                features.shape[0], dtype=torch.int64
            ),  # @TODO: Necessary?
            features=self._features,
        )

        # @TODO: This sort of hard-coding is not ideal; all features should be
        #        captured by `FEATURES` and included in the output of
        #        `I3FeatureExtractor`.
        data.n_pulses = n_pulses
        return data

    def _extract_feature_array_from_frame(self, frame: I3Frame) -> np.array:
        features = None
        for i3extractor in self._i3_extractor:
            feature_dict = i3extractor(frame)
            features_pulsemap = np.array(
                [feature_dict[key] for key in self._features]
            ).T
            if features is None:
                features = features_pulsemap
            else:
                features = np.concatenate(
                    (features, features_pulsemap), axis=0
                )
        return features

    def _submit_predictions(
        self, frame: I3Frame, predictions: np.ndarray
    ) -> I3Frame:
        # RunID       = frame['I3EventHeader'].run_id
        # SubrunID    = frame['I3EventHeader'].sub_run_id
        # EventID     = frame['I3EventHeader'].event_id
        # SubEventID  = frame['I3EventHeader'].sub_event_id
        # MCInIcePrimary = get_primary_particle(frame)
        # energy  =  MCInIcePrimary.energy
        # azimuth =  MCInIcePrimary.dir.azimuth
        # zenith  =  MCInIcePrimary.dir.zenith
        # pid = MCInIcePrimary.pdg_encoding

        # predictions, event_no = get_predictions(RunID, SubrunID, EventID, SubEventID, energy, azimuth, zenith, pid,con)
        pulsemap = dataclasses.I3RecoPulseSeriesMap.from_frame(
            frame, self._pulsemap
        )

        idx = 0
        predictions_map = dataclasses.I3MapKeyVectorDouble()
        for om_key, pulses in pulsemap.items():
            num_pulses = len(pulses)
            predictions_map[om_key] = predictions[
                idx : idx + num_pulses
            ].tolist()
            idx += num_pulses

        assert idx == len(
            predictions
        ), "Not all predictions were mapped to pulses"
        frame.Put(self._predictions_key, predictions_map)

        # Checks
        assert (
            pulsemap.keys() == predictions_map.keys()
        ), "Input pulse map and predictions map do not contain exactly the same OMs"
        # Create a pulse map mask, indicating the pulses that are over threshold (e.g. identified as signal) and therefore should be kept
        # Using a lambda function to evaluate which pulses to keep by checking the prediction for each pulse
        frame.Put(
            self._total_pulsemap_name,
            dataclasses.I3RecoPulseSeriesMapMask(
                frame,
                self._pulsemap,
                lambda om_key, index, pulse: predictions_map[om_key][index]
                >= self._threshold,
            ),
        )
        return frame

    def _add_meta_data(self, frame: I3Frame) -> I3Frame:
        doc_url = "https://github.com/graphnet-team/analyses/tree/main/upgrade_noise_cleaning"
        # event_no = meta_data[0][0]
        # was_training_event = meta_data[0][1]
        # was_validation_event = meta_data[0][2]
        # was_test_event = meta_data[0][3]
        # meta data
        # frame.Put('graphnet_event_no', icetray.I3Int(event_no))
        frame.Put("graphnet_docs", dataclasses.I3String(doc_url))
        # frame.Put('graphnet_was_training_event', icetray.I3Bool(bool(was_training_event)))
        # frame.Put('graphnet_was_validation_event', icetray.I3Bool(bool(was_validation_event)))
        # frame.Put('graphnet_was_test_event', icetray.I3Bool(bool(was_test_event)))
        return frame

    def _add_pulsemap_for_each_dom_type(
        self, frame: I3Frame, gcd_file: Any
    ) -> I3Frame:
        g = dataio.I3File(gcd_file)
        gFrame = g.pop_frame()
        while "I3Geometry" not in gFrame.keys():
            gFrame = g.pop_frame()
        omGeoMap = gFrame["I3Geometry"].omgeo

        mDOMMap, DEggMap, IceCubeMap = {}, {}, {}
        pulses = dataclasses.I3RecoPulseSeriesMap.from_frame(
            frame, self._total_pulsemap_name
        )
        for P in pulses:
            om = omGeoMap[P[0]]
            if om.omtype == 130:  # "mDOM"
                mDOMMap[P[0]] = P[1]
            elif om.omtype == 120:  # "DEgg"
                DEggMap[P[0]] = P[1]
            elif om.omtype == 20:  # "IceCube"
                IceCubeMap[P[0]] = P[1]

        frame.Put(
            f"{self._total_pulsemap_name}_mDOMs_Only",
            dataclasses.I3RecoPulseSeriesMap(mDOMMap),
        )
        frame.Put(
            f"{self._total_pulsemap_name}_dEggs_Only",
            dataclasses.I3RecoPulseSeriesMap(DEggMap),
        )
        frame.Put(
            f"{self._total_pulsemap_name}_pDOMs_Only",
            dataclasses.I3RecoPulseSeriesMap(IceCubeMap),
        )
        return frame


class GraphNeTModuleIceCube86(GraphNeTModuleBase):
    """Module for running GraphNeT models on standard IceCube-86 data."""

    FEATURES = FEATURES.ICECUBE86
    I3FEATUREEXTRACTOR_CLASS = I3FeatureExtractorIceCube86


class GraphNeTModuleIceCubeDeepCore(GraphNeTModuleBase):
    """Module for running GraphNeT models on standard IceCube-DeepCore data."""

    FEATURES = FEATURES.DEEPCORE
    I3FEATUREEXTRACTOR_CLASS = I3FeatureExtractorIceCubeDeepCore


class GraphNeTModuleIceCubeUpgrade(GraphNeTModuleBase):
    """Module for running GraphNeT models on standard IceCube-Upgrade data."""

    FEATURES = FEATURES.UPGRADE
    I3FEATUREEXTRACTOR_CLASS = I3FeatureExtractorIceCubeUpgrade
