__all__ = ["plot", "plot_locations", "Kuramoto", "Fridlund"]

import abc
import datetime
import json
import pathlib
from typing import Optional

import cv2
import numpy as np
from scipy import interpolate

from . import consts

DEFAULT_SIZE_PX = 4096


class Scheme(abc.ABC):
    """Scheme for plotting the EMG values on a 2D canvas.

    The EMG values are plotted on a 2D canvas. The canvas is a numpy array
    and the locations of the EMG values are defined by the concrete scheme implementation.

    The vertical axis is the y-axis and the horizontal axis is the x-axis.
    The vertical axis the symmetry axis of the face (x=0).
    The horizontal axis is the "Frankfort horizontal plane" (y=0).

    """

    def __init__(self) -> None:
        self.locations: dict[str, tuple[int, int]] = self.load_locs()
        self.outer_hull = cv2.convexHull(consts.FACE_COORDS, returnPoints=True).reshape(-1, 2)
        self.outer_hull = np.array([abs_to_rel(x, y, (4096, 4096)) for x, y in self.outer_hull])
        self.outer_dict = {f"O{i}": (x, y) for i, (x, y) in enumerate(self.outer_hull)}

        self.__lut: dict[str, np.ndarray] = {}

    def check(self, emg_values: dict[str, float]) -> bool:
        # check if all values are inside the dict
        # and if they are in the correct range
        temp_locations = self.locations.copy()
        for emg_name, emg_value in emg_values.items():
            if emg_name not in temp_locations:
                return False
            if not self._check_value(emg_value):
                return False
            del temp_locations[emg_name]
        return True

    def _check_value(self, emg_value: float) -> bool:
        return emg_value >= 0

    @abc.abstractmethod
    def save_locs(self) -> None:
        pass

    @abc.abstractmethod
    def load_locs(self) -> None:
        pass


class Kuramoto(Scheme):
    def save_locs(self) -> None:
        p = pathlib.Path(__file__).parent / "locations_kuramoto.json"
        p.write_text(json.dumps(self.locations, indent=4))

        # save the current version of the file
        p = pathlib.Path(__file__).parent / f"locations_kuramoto_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        p.write_text(json.dumps(self.locations, indent=4))

    def load_locs(self) -> None:
        p = pathlib.Path(__file__).parent / "locations_kuramoto.json"
        return json.loads(p.read_text())


class Fridlund(Scheme):
    def save_locs(self) -> None:
        p = pathlib.Path(__file__).parent / "locations_fridlund.json"
        p.write_text(json.dumps(self.locations, indent=4))

        # save the current version of the file
        p = pathlib.Path(__file__).parent / f"locations_fridlund_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        p.write_text(json.dumps(self.locations, indent=4))

    def load_locs(self) -> None:
        p = pathlib.Path(__file__).parent / "locations_fridlund.json"
        return json.loads(p.read_text())


def rel_to_abs(rel_x: float, rel_y: float, size: tuple[int, int]) -> tuple[int, int]:
    # the values are relative to the center of the canvas
    # thus are in the range [-100, 100] and need to be scaled accordingly
    # but the y-axis is flipped, thus the y-value needs to be inverted
    abs_x = int((rel_x + 100) / 200 * size[0])
    abs_y = int((100 - rel_y) / 200 * size[1])

    return abs_x, abs_y


def abs_to_rel(abs_x: int, abs_y: int, size: tuple[int, int]) -> tuple[float, float]:
    rel_x = (abs_x / size[0] * 200) - 100
    rel_y = 100 - (abs_y / size[1] * 200)

    return rel_x, rel_y


def plot_locations(
    canvas: Optional[np.ndarray],
    scheme: Scheme,
    draw_outer_hull: bool = True,
    shape: tuple[int, int] = (256, 256),
):
    if canvas is None:
        canvas = np.zeros((*shape, 3), dtype=np.float32)

    for emg_name, emg_loc in scheme.locations.items():
        x, y = rel_to_abs(emg_loc[0], emg_loc[1], canvas.shape)
        cv2.circle(canvas, (x, y), 50, (0, 0, 255), -1)

        text_size, _ = cv2.getTextSize(str(emg_name), cv2.FONT_HERSHEY_SIMPLEX, 2, 8)
        text_x = x - text_size[0] // 2
        text_y = y + text_size[1] // 2

        cv2.putText(canvas, str(emg_name), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 8)

    if draw_outer_hull:
        for name, loc in scheme.outer_dict.items():
            x, y = rel_to_abs(loc[0], loc[1], canvas.shape)
            cv2.circle(canvas, (x, y), 50, (0, 255, 0), -1)

            text_size, _ = cv2.getTextSize(str(name), cv2.FONT_HERSHEY_SIMPLEX, 2, 8)
            text_x = x - text_size[0] // 2
            text_y = y + text_size[1] // 2

            cv2.putText(canvas, str(name), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 8)

    return canvas


def plot(
    canvas: Optional[np.ndarray],
    scheme: Scheme,
    emg_values: dict[str, float],
    shape: tuple[int, int] = (1024, 1024),
) -> np.ndarray:
    if canvas is None:
        canvas = np.zeros((*shape, 1), dtype=np.float32)

    keys_sorted_semg = sorted(scheme.locations.keys())
    keys_sorted_hull = sorted(scheme.outer_dict.keys())

    # # get the values for each location
    xy = np.array([scheme.locations[k] for k in keys_sorted_semg] + [scheme.outer_dict[k] for k in keys_sorted_hull])
    v = np.array([emg_values[k] for k in keys_sorted_semg] + [0] * len(keys_sorted_hull))

    X = np.linspace(xy.min(axis=0)[0], xy.max(axis=0)[0], canvas.shape[0])
    Y = np.linspace(xy.min(axis=0)[1], xy.max(axis=0)[1], canvas.shape[1])
    Y = np.flip(Y)
    X, Y = np.meshgrid(X, Y)

    interp = interpolate.CloughTocher2DInterpolator(xy, v)
    # interp = interpolate.LinearNDInterpolator(xy, v)
    # interp = interpolate.NearestNDInterpolator(xy, v)

    Z = interp(X, Y)
    Z = (Z - np.nanmin(Z)) / (np.nanmax(Z) - np.nanmin(Z)) * 255
    Z = Z.astype(np.uint8)
    Z = cv2.applyColorMap(Z, cv2.COLORMAP_VIRIDIS)
    Z = cv2.cvtColor(Z, cv2.COLOR_BGR2RGB)
    return Z
