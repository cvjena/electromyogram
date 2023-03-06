__all__ = ["interpolate", "plot_locations", "Kuramoto", "Fridlund", "colorize", "get_colormap"]

import abc
import datetime
import json
import pathlib
from typing import Optional, Type, Union

import cv2
import numpy as np
from scipy import interpolate as interp

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

    def valid(self, emg_values: dict[str, float]) -> bool:
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
    pairs_L = ["E1", "E3", "E5", "E7", "E9", "E13", "E15", "E17"]
    pairs_R = ["E2", "E4", "E6", "E8", "E10", "E14", "E16", "E18"]

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
    pairs_L = ["Dao li", "OrbOr li", "Ment li", "Mass li", "Zyg li", "Llsup li", "OrbOc li", "lat Front li", "med Front li", "Corr li", "Deprsup li"]
    pairs_R = ["Dao re", "OrbOr re", "Ment re", "Mass re", "Zyg re", "Llsup re", "OrbOc re", "lat Front re", "med Front re", "Corr re", "Deprsup re"]

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
    scheme: Scheme,
    shape: tuple[int, int] = (512, 512),
    draw_outer_hull: bool = True,
) -> np.ndarray:
    """Plot the locations of the EMG values on a 2D canvas.

    This function is used to plot the locations of the EMG values on a 2D canvas.
    It visualizes the locations of the EMG values and the outer hull of the face.

    Parameters
    ----------
    scheme : Scheme
        The scheme to use for plotting the EMG values.
    shape : tuple[int, int], optional
        The shape of the canvas, by default (512, 512)
    draw_outer_hull : bool, optional
        Whether to draw the outer hull of the face, by default True

    Returns
    -------
    np.ndarray
        The canvas with the plotted EMG values.
    """
    canvas = np.zeros((*shape, 3), dtype=np.uint8)

    scale_factor = shape[0] / DEFAULT_SIZE_PX

    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 2 * scale_factor
    thickness = int(8 * scale_factor)
    radius = int(50 * scale_factor)

    def _draw(img: np.ndarray, text: str, x: int, y: int, color: tuple = (0, 0, 255)):
        cv2.circle(img, (x, y), radius, color, -1)
        text_size, _ = cv2.getTextSize(text, fontFace, fontScale, thickness)
        text_x = x - text_size[0] // 2
        text_y = y + text_size[1] // 2
        cv2.putText(img, text, (text_x, text_y), fontFace, fontScale, (255, 255, 255), thickness)

    for emg_name, emg_loc in scheme.locations.items():
        x, y = rel_to_abs(emg_loc[0], emg_loc[1], canvas.shape)
        _draw(canvas, emg_name, x, y)

    if draw_outer_hull:
        for name, loc in scheme.outer_dict.items():
            x, y = rel_to_abs(loc[0], loc[1], canvas.shape)
            _draw(canvas, name, x, y, color=(0, 255, 0))

    return canvas


def interpolate(
    scheme: Scheme,
    emg_values: dict[str, float],
    shape: tuple[int, int] = (512, 512),
    vmin: float = 0.0,
    vmax: Optional[float] = None,
    mirror: bool = False,
    mirror_plane_width: int = 2,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if not mirror:
        return __interpolate(scheme, emg_values, shape, vmin, vmax)

    emg_values_mirrored_l, emg_values_mirrored_r = scheme.mirror(emg_values)
    interpolation_n = __interpolate(scheme, emg_values, shape, vmin, vmax)
    interpolation_l = __interpolate(scheme, emg_values_mirrored_l, shape, vmin, vmax)
    interpolation_r = __interpolate(scheme, emg_values_mirrored_r, shape, vmin, vmax)

    # draw a vertical line in the mirrored images to indicate the mirror plane
    middle_slice = slice(shape[1] // 2 - mirror_plane_width, shape[1] // 2 + mirror_plane_width)
    interpolation_l[:, middle_slice] = 0
    interpolation_r[:, middle_slice] = 0

    return interpolation_n, interpolation_l, interpolation_r


def __interpolate(
    scheme: Scheme,
    emg_values: dict[str, float],
    shape: tuple[int, int] = (512, 512),
    vmin: float = 0.0,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Interpolate the EMG values to a 2D canvas.

    This function is used to interpolate the EMG values to a 2D canvas based on the given scheme.
    We use a RBF interpolation to interpolate the EMG values to the canvas.

    Parameters
    ----------
    scheme : Scheme
        The scheme to use for interpolation.
        Currently only the `Kuramoto` und `Fridlund` scheme are supported.
    emg_values : dict[str, float]
        The EMG values to interpolate.
        The keys of the dict need to match the keys of the scheme.
    shape : tuple[int, int], optional
        The shape of the canvas, by default (512, 512)
    vmin : float
        The minimum value of the EMG values. Defaults to 0.0.
    vmax : Optional[float]
        The maximum value of the EMG values. Defaults to None and will be set to the maximum value of the EMG values.
    """

    if not scheme.valid(emg_values):
        raise ValueError("Either missing or invalid EMG keys/values in dict")

    canvas = np.zeros(shape, dtype=np.float32)
    keys_sorted_semg = sorted(scheme.locations.keys())
    keys_sorted_hull = sorted(scheme.outer_dict.keys())

    # # get the values for each location
    xy = np.array([scheme.locations[k] for k in keys_sorted_semg] + [scheme.outer_dict[k] for k in keys_sorted_hull])
    v = np.array([emg_values[k] for k in keys_sorted_semg] + [0] * len(keys_sorted_hull))

    vmin = vmin or v.min()
    vmax = vmax or v.max()

    # prepare the data for RBF interpolation
    p = xy.reshape(-1, 2)
    v = v.reshape(-1, 1)
    x_grid = np.mgrid[-100 : 100 : canvas.shape[0] * 1j, -100 : 100 : canvas.shape[1] * 1j].reshape(2, -1).T
    Z = interp.RBFInterpolator(p, v, kernel="thin_plate_spline", smoothing=0.1)(x_grid)
    # reshape the data to the correct shape, and transpose it such it is rotated 90 degrees counter-clockwise
    Z = np.rot90(Z.reshape(canvas.shape[0], canvas.shape[1]))
    # all values smaller than 0 are set to 0
    Z[Z < vmin] = vmin
    Z = (Z - vmin) / (Z.max() - vmin)  # normalize the values to the range [0, 1]
    return Z * vmax


def get_colormap(
    cmap_name: Union[str, object],
) -> np.ndarray:
    try:
        import matplotlib
    except ImportError:
        raise ImportError("matplotlib must be installed to use this function")

    if isinstance(cmap_name, str):
        if cmap_name in matplotlib.colormaps:
            # this is a bit janky, as matplotlib.colormaps is actually deprecated
            colormap = matplotlib.cm.get_cmap(cmap_name)
        elif cmap_name == "parula":
            colormap = matplotlib.colors.LinearSegmentedColormap.from_list("parula", consts.parula_colormap)
        else:
            raise ValueError(f"{cmap_name} is not a valid colormap name")

    elif isinstance(cmap_name, object):
        try:
            # this should work for palettable colormaps
            # as we have it as a dependency
            colormap = cmap_name.mpl_colormap
        except ImportError:
            raise ImportError("palettable must be installed to use this function")
        except Exception:
            raise ValueError(f"{cmap_name} is not a valid colormap class in palettable")
    else:
        raise ValueError(f"{cmap_name} does not have a valid colormap name or class")
    return colormap


def apply_colormap(electromyogram: np.ndarray, colormap) -> np.ndarray:
    # TODO later support alpha channel
    return (colormap(electromyogram)[..., :3] * 255).astype(np.uint8)


def colorize(
    interpolation: np.ndarray,
    cmap: Optional[Union[str, Type]] = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Colorize an electromyogram interpolation using a given colormap

    The colormap can be either a string or an int. If it is a string, it must be a valid
    supported colormap name. If it is an int, it must be a valid OpenCV colormap code.
    eg. cv2.COLORMAP_VIRIDIS

    Args:
        electromyogram (np.ndarray): The electromyogram interpolation to colorize
        colormap (Union[int, str], optional): The colormap to use. Defaults to "viridis".
        vmin (Optional[float]): The minimum value of the electromyogram, defaults to the minimum value of the electromyogram
        vmax (Optional[float]): The maximum value of the electromyogram, defaults to the maximum value of the electromyogram

    Raises:
        ValueError: If the electromyogram is not 2-dimensional
        ValueError: If the colormap is neither an int nor a string
        ValueError: If the electromyogram values are outside of the range [vmin, vmax]

    Returns:
        np.ndarray: The colorized electromyogram (np.uint8)
    """

    if interpolation.ndim == 3:
        raise ValueError("electromyogram must be 2-dimensional")

    if not isinstance(cmap, str) and not isinstance(cmap, object):
        raise ValueError(f"colormap must be either a string and not {type(cmap)}")

    if vmin is None:
        vmin = np.min(interpolation)
    if vmax is None:
        vmax = np.max(interpolation)

    if np.max(interpolation) > vmax or np.min(interpolation) < vmin:
        raise ValueError("electromyogram values are outside of the range [vmin, vmax]")

    # normalize the values between vmin and vmax
    interpolation = (interpolation - vmin) / (vmax - vmin)

    # scale the values to the range [0, 255]
    interpolation = (interpolation * 255).astype(np.uint8)

    return apply_colormap(interpolation, get_colormap(cmap))
