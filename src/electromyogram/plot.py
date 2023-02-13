__all__ = ["plot_locations", "Kuramoto", "Fridlund"]

import abc
import datetime
import json
import pathlib
from typing import Optional

import cv2
import numpy as np

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
        self.locations = self.load_locs()

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


def plot_locations(
    canvas: Optional[np.ndarray],
    scheme: Scheme,
):
    if canvas is None:
        canvas = np.zeros((DEFAULT_SIZE_PX, DEFAULT_SIZE_PX, 3), dtype=np.float32)

    for emg_name, emg_loc in scheme.locations.items():
        x, y = rel_to_abs(emg_loc[0], emg_loc[1], canvas.shape)
        cv2.circle(canvas, (x, y), 50, (0, 0, 255), -1)

        text_size, _ = cv2.getTextSize(str(emg_name), cv2.FONT_HERSHEY_SIMPLEX, 2, 8)
        text_x = x - text_size[0] // 2
        text_y = y + text_size[1] // 2

        cv2.putText(canvas, str(emg_name), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 8)

    return canvas
