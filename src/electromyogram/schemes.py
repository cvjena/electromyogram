__all__ = ["Scheme", "Kuramoto", "Fridlund"]

import abc
from typing import Optional

import cv2
import numpy as np

from . import consts
from .utils import abs_to_rel


class Scheme(abc.ABC):
    """Scheme for plotting the EMG values on a 2D canvas.

    The EMG values are plotted on a 2D canvas. The canvas is a numpy array
    and the locations of the EMG values are defined by the concrete scheme implementation.

    The vertical axis is the y-axis and the horizontal axis is the x-axis.
    The vertical axis the symmetry axis of the face (x=0).
    The horizontal axis is the "Frankfort horizontal plane" (y=0).

    """

    pairs_L: Optional[list[str]] = None
    pairs_R: Optional[list[str]] = None
    mapping: dict[str, str] = None

    locations: dict[str, tuple[float, float]] = None

    def __init__(self) -> None:
        if self.locations is None:
            raise ValueError("Locations have to be implemented by the sub classes.")
        
        # TODO we should only do this is they want to mirror the values
        if self.pairs_L is None:
            raise ValueError("Pair values have to be implemented by the sub classes.")
        if self.pairs_R is None:
            raise ValueError("Pair values have to be implemented by the sub classes.")
        if len(self.pairs_L) != len(self.pairs_R):
            raise ValueError("Number of pairs have to be the same.")
        if self.mapping is None:
            raise ValueError("Shortcuts have to be implemented by the sub classes.")
        
        self.outer_hull = cv2.convexHull(consts.FACE_COORDS, returnPoints=True).reshape(-1, 2)
        self.outer_hull = np.array([abs_to_rel(x, y, (4096, 4096)) for x, y in self.outer_hull])
        self.outer_dict = {f"O{i}": (x, y) for i, (x, y) in enumerate(self.outer_hull)}

    def validify(
        self, 
        emg_values: dict[str, float],
        missing_value: float = 0.0,
    ) -> dict[str, float]:
        """
        This function checks if the given emg key+value pairs are valid.
        It attempts to fix replace the key by checking if the key is in the locations,
        or part of the known mappings!
        """
        valid_dict: dict[str, tuple[float, tuple[float, float]]] = {}
        
        for emg_name, emg_value in emg_values.items():
            if not self._check_value(emg_value):
                raise ValueError(f"EMG value {emg_value} for {emg_name} is not valid. (smaller than 0)")

            if emg_name not in self.locations and emg_name not in self.mapping:
                raise ValueError(f"EMG name {emg_name} is not valid (in locations and mappings).")
            
            key = emg_name if emg_name in self.locations else self.mapping[emg_name]
            valid_dict[key] = emg_value, self.locations[key]
            
        # add the missing values
        for key in self.locations.keys():
            if key not in valid_dict:
                valid_dict[key] = missing_value, self.locations[key]
        return valid_dict

    def _check_value(self, emg_value: float) -> bool:
        return emg_value >= 0

    def mirror(self, emg_values: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
        """Mirror the EMG values.

        This function mirrors the EMG values.
        It creates two new dicts, one for the left side and one for the right side of the face.
        The dict for left then contains the EMG values for the left side of the face.
        The dict for right then contains the EMG values for the right side of the face.
        """
        assert self.pairs_L is not None, "Pair values have to be implemented by the sub classes."
        assert self.pairs_R is not None, "Pair values have to be implemented by the sub classes."
        assert len(self.pairs_L) == len(self.pairs_R)

        vals_left = {}
        # mirror the values, i.e. swap the left and right values
        for emg_name, emg_value in emg_values.items():
            if emg_name in self.pairs_R:
                name_left = self.pairs_L[self.pairs_R.index(emg_name)]
                left_val = emg_values[name_left]
                vals_left[emg_name] = left_val
            else:
                vals_left[emg_name] = emg_value

        vals_right = {}
        # mirror the values, i.e. swap the left and right values
        for emg_name, emg_value in emg_values.items():
            if emg_name in self.pairs_L:
                name_right = self.pairs_R[self.pairs_L.index(emg_name)]
                right_val = emg_values[name_right]
                vals_right[emg_name] = right_val
            else:
                vals_right[emg_name] = emg_value

        return vals_right, vals_left


class Kuramoto(Scheme):
    # given as (L, R)
    pairs_L = ["K2", "K4", "K6", "K8", "K10", "K14", "K16", "K18"]
    pairs_R = ["K1", "K3", "K5", "K7",  "K9", "K13", "K15", "K17"]
    # fmt: off
    locations = {
        "K1":  [-20.0,  66.0],
        "K2":  [ 20.0,  66.0],
        "K3":  [-32.0,  56.0],
        "K4":  [ 32.0,  56.0],
        "K5":  [-44.0,  13.0],
        "K6":  [ 44.0,  13.0],
        "K7":  [-25.0,  -9.0],
        "K8":  [ 25.0,  -9.0],
        "K9":  [-20.0, -67.0],
        "K10": [ 20.0, -67.0],
        "K13": [-63.0,  40.0],
        "K14": [ 63.0,  40.0],
        "K15": [-58.0,   6.0],
        "K16": [ 59.0,   6.0],
        "K17": [-72.0, -50.0],
        "K18": [ 72.0, -50.0],
        "K19": [  0.0,  47.0],
        "K20": [  0.0, -26.0],
        "K24": [  0.0,  26.0],
    }
    # fmt: on
    mapping = {n.replace("K", "E") : n for n in locations.keys()}

class Fridlund(Scheme):
    mapping = {
        "Corr li": "F6",
        "Corr re": "F5",
        "DAO li": "F14",
        "DAO re": "F13",
        "Deprsup li": "F8",
        "Deprsup re": "F7",
        "lat Front li": "F4",
        "lat Front re": "F3",
        "Llsup li": "F10",
        "Llsup re": "F9",
        "Mass li": "F22",
        "Mass re": "F21",
        "med Front li": "F2",
        "med Front re": "F1",
        "Ment li": "F16",
        "Ment re": "F15",
        "OrbOc li": "F18",
        "OrbOc re": "F17",
        "OrbOr li": "F12",
        "OrbOr re": "F11",
        "Zyg li": "F20",
        "Zyg re": "F19",
    }
    pairs_L = ["F14", "F12", "F16", "F22", "F20", "F10", "F18", "F4", "F2", "F6", "F8"]
    pairs_R = ["F13", "F11", "F15", "F21", "F19",  "F9", "F17", "F3", "F1", "F5", "F7"]

    # fmt: off
    locations = {
        "F14": [ 26.0, -69.0],
        "F13": [-26.0, -69.0],
        "F12": [ 21.0, -52.0],
        "F11": [-21.0, -52.0],
        "F16": [  6.0, -65.0],
        "F15": [ -6.0, -65.0],
        "F22": [ 70.0, -50.0],
        "F21": [-70.0, -50.0],
        "F20": [ 53.0, -23.0],
        "F19": [-53.0, -23.0],
        "F10": [ 28.0, -12.0],
         "F9": [-28.0, -12.0],
        "F18": [ 42.0,  10.0],
        "F17": [-42.0,  10.0],
         "F4": [ 32.0,  60.0],
         "F3": [-32.0,  60.0],
         "F2": [ 10.0,  65.0],
         "F1": [-10.0,  65.0],
         "F6": [ 22.0,  52.0],
         "F5": [-22.0,  52.0],
         "F8": [  9.0,  49.0],
         "F7": [ -9.0,  49.0],
    }
    # fmt: on