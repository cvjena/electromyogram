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
    shortcuts: dict[str, str] = None

    locations: dict[str, tuple[float, float]] = None

    def __init__(self) -> None:
        if self.locations is None:
            raise ValueError("Locations have to be implemented by the sub classes.")
        if self.pairs_L is None:
            raise ValueError("Pair values have to be implemented by the sub classes.")
        if self.pairs_R is None:
            raise ValueError("Pair values have to be implemented by the sub classes.")
        if len(self.pairs_L) != len(self.pairs_R):
            raise ValueError("Number of pairs have to be the same.")
        if self.shortcuts is None:
            raise ValueError("Shortcuts have to be implemented by the sub classes.")
        
        self.outer_hull = cv2.convexHull(consts.FACE_COORDS, returnPoints=True).reshape(-1, 2)
        self.outer_hull = np.array([abs_to_rel(x, y, (4096, 4096)) for x, y in self.outer_hull])
        self.outer_dict = {f"O{i}": (x, y) for i, (x, y) in enumerate(self.outer_hull)}

    def valid(self, emg_values: dict[str, float]) -> bool:
        # check if all values are inside the dict
        # and if they are in the correct range
        temp_locations = self.locations.copy()
        for emg_name, emg_value in emg_values.items():
            if emg_name not in temp_locations:
                raise ValueError(f"EMG name {emg_name} not in locations.")
            if not self._check_value(emg_value):
                raise ValueError(f"EMG value {emg_value} for {emg_name} is not valid.")
            del temp_locations[emg_name]

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
    pairs_L = ["E2", "E4", "E6", "E8", "E10", "E14", "E16", "E18"]
    pairs_R = ["E1", "E3", "E5", "E7", "E9", "E13", "E15", "E17"]
    shortcuts = {n: n.replace("E", "K") for n in pairs_L + pairs_R}

    # fmt: off
    locations = {
        "E1":  [-20.0,  66.0],
        "E2":  [ 20.0,  66.0],
        "E3":  [-32.0,  56.0],
        "E4":  [ 32.0,  56.0],
        "E5":  [-44.0,  13.0],
        "E6":  [ 44.0,  13.0],
        "E7":  [-25.0,  -9.0],
        "E8":  [ 25.0,  -9.0],
        "E9":  [-20.0, -67.0],
        "E10": [ 20.0, -67.0],
        "E13": [-63.0,  40.0],
        "E14": [ 63.0,  40.0],
        "E15": [-58.0,   6.0],
        "E16": [ 59.0,   6.0],
        "E17": [-72.0, -50.0],
        "E18": [ 72.0, -50.0],
        "E19": [  0.0,  47.0],
        "E20": [  0.0, -26.0],
        "E24": [  0.0,  26.0],
    }
    # fmt: on


class Fridlund(Scheme):
    pairs_L = ["DAO li", "OrbOr li", "Ment li", "Mass li", "Zyg li", "Llsup li", "OrbOc li", "lat Front li", "med Front li", "Corr li", "Deprsup li"]
    pairs_R = ["DAO re", "OrbOr re", "Ment re", "Mass re", "Zyg re", "Llsup re", "OrbOc re", "lat Front re", "med Front re", "Corr re", "Deprsup re"]
    shortcuts = {
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
    # fmt: off
    locations = {
        "DAO li": [26.0, -69.0],
        "DAO re": [-26.0, -69.0],
        "OrbOr li": [21.0, -52.0],
        "OrbOr re": [-21.0, -52.0],
        "Ment li": [6.0, -65.0],
        "Ment re": [-6.0, -65.0],
        "Mass li": [70.0, -50],
        "Mass re": [-70.0, -50],
        "Zyg li": [53.0, -23],
        "Zyg re": [-53.0, -23],
        "Llsup li": [28.0, -12.0],
        "Llsup re": [-28.0, -12.0],
        "OrbOc li": [42.0, 10.0],
        "OrbOc re": [-42.0, 10.0],
        "lat Front li": [32.0, 60.0],
        "lat Front re": [-32.0, 60.0],
        "med Front li": [10, 65.0],
        "med Front re": [-10, 65.0],
        "Corr li": [22.0, 52.0],
        "Corr re": [-22.0, 52.0],
        "Deprsup li": [9.0, 49.0],
        "Deprsup re": [-9.0, 49.0],
    }
    # fmt: on