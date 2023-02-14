__all__ = ["plot", "plot_locations", "Kuramoto", "Fridlund"]

import abc
import datetime
import json
import pathlib
from typing import Optional

import cv2
import numpy as np

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

    def _compute_nn(self, loc: tuple[int, int], neighbours: int, options: np.ndarray) -> np.ndarray:
        """Compute the nearest neighbours for a given location.

        Args:
            loc (tuple[int, int]): The location for which to compute the nearest neighbours.
            neighbours (int): The number of nearest neighbours to compute.
        """

        # compute the distance to all locations
        distances = np.linalg.norm(options - np.array(loc), axis=1)
        # sort the distances and return the indices of the smallest ones
        sort = np.argsort(distances)[:neighbours]
        return sort, distances[sort]

    def compute_nn(self, img_shape: tuple[int, int], neighbours: int = 4) -> None:
        """Compute the nearest neighbours for each location in the scheme.

        This function computes for each pixel the accoring nearest neighbour based
        on the scheme locations and the face coordinates. It then saves the result
        inside a numpy array with the same shape as the image but with the n channels.
        This way the nearest neighbour can be computed in constant time.

        Args:
            img_shape (tuple[int, int]): The shape of the image.
            neighbours (int): The number of nearest neighbours to compute.
        """

        # check if the look up table has already been computed
        lut_hash = f"{img_shape}-{neighbours}"

        if lut_hash in self.__lut:
            return self.__lut[lut_hash]

        keys_sorted_semg = sorted(self.locations.keys())
        keys_sorted_hull = sorted(self.outer_dict.keys())

        options_semg = np.array([rel_to_abs(*self.locations[k], img_shape) for k in keys_sorted_semg])
        options_hull = np.array([rel_to_abs(*self.outer_dict[k], img_shape) for k in keys_sorted_hull])

        options = np.concatenate((options_semg, options_hull), axis=0)

        look_up_table_idx = np.zeros((img_shape[0], img_shape[1], neighbours), dtype=np.int8)
        look_up_table_dis = np.zeros((img_shape[0], img_shape[1], neighbours), dtype=np.float32)

        # apply the nearest neighbour algorithm to each pixel in the image
        # and do it in parallel
        # TODO to this in parallel and faster
        for i, j in np.ndindex(img_shape):
            # compute the nearest neighbours for the current pixel
            # and save them in the look up table
            idx, dist = self._compute_nn((i, j), neighbours, options)
            look_up_table_idx[i, j] = idx
            look_up_table_dis[i, j] = dist

        self.__lut[lut_hash] = look_up_table_idx, look_up_table_dis
        return look_up_table_idx, look_up_table_dis


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
    neighbours: int = 4,
    shape: tuple[int, int] = (512, 512),
) -> np.ndarray:
    if canvas is None:
        canvas = np.zeros((*shape, 1), dtype=np.float32)

    # compute the nearest neighbours for each location
    lup_idx, lup_dis = scheme.compute_nn(canvas.shape[:2], neighbours=neighbours)

    keys_sorted_semg = sorted(scheme.locations.keys())
    keys_sorted_hull = sorted(scheme.outer_dict.keys())

    # get the values for each location
    values = np.array([emg_values[k] for k in keys_sorted_semg] + [0] * len(keys_sorted_hull))
    # TODO skip the outer hull

    for x, y in np.ndindex(canvas.shape[:2]):
        # get the nearest neighbours for the current pixel
        nn = lup_idx[y, x]
        nd = lup_dis[y, x] + 1e-6
        # compute the average value of the nearest neighbours
        avg = np.average(values[nn], weights=1 / nd)
        # set the pixel to the average value
        canvas[x, y] = avg

    # use only the area inside the outer hull
    outer_hull_abs = [rel_to_abs(x, y, canvas.shape) for x, y in scheme.outer_hull]
    mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
    mask = cv2.fillPoly(mask, [np.array(outer_hull_abs)], (255, 255, 255))
    canvas = cv2.bitwise_and(canvas, canvas, mask=mask)
    canvas = (canvas - np.min(canvas)) / (np.max(canvas) - np.min(canvas))
    return canvas
