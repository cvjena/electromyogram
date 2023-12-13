__all__ = ["interpolate", "plot_locations", "colorize", "get_colormap", "postprocess"]


import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Type, Union

import cv2
import h5py
import numpy as np
from scipy import interpolate as interp

from .consts import parula_colormap
from .schemes import Scheme
from .utils import rel_to_abs


@dataclass
class FaceModel:
    points: np.ndarray
    triangles: np.ndarray
    facets: np.ndarray
    masking: np.ndarray
    
    def get_outer(self, shape:tuple[int, int]) -> np.ndarray:
        # points = np.array([rel_to_abs(x, y, shape) for x, y in ])
        points = np.copy(self.points)
        points[:, 0] *= shape[0]
        points[:, 1] *= shape[1]
        points = points.astype(np.int32)
        return cv2.convexHull(points, returnPoints=True).reshape(-1, 2)

face_model_data = h5py.File(Path(__file__).parent / "face_model.h5", "r")

face_model = FaceModel(
    points=np.array(face_model_data["points"]),
    triangles=np.array(face_model_data["triangles"]),
    facets=np.array(face_model_data["facets"]),
    masking=np.array(face_model_data["masking_canonical"]),
)

face_model_data.close()

def plot_locations(
    scheme: Scheme,
    shape: tuple[int, int] = (512, 512),
    fontScale: float = 0.5,
    thickness: int = 2,
    radius: int = 7,
    color_circle: tuple[int, int, int] = (255, 105, 180),
    do_postprocess: bool = True,
    draw_outerhull: bool = False,
    canvas: Optional[np.ndarray] = None,
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
    canvas : Optional[np.ndarray], optional
        The canvas to draw on, by default None

    Returns
    -------
    np.ndarray
        The canvas with the plotted EMG values.
    """
    if canvas is None:
        canvas = np.full((shape[0], shape[1], 3), fill_value=255, dtype=np.uint8)
    if do_postprocess:
        canvas = postprocess(canvas, remove_outer=True, draw_triangle=True, invert=True)

    scale_factor = shape[0] / 512 # all values are relative to a 512x512 canvas and optimized for that size

    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    lineType = cv2.LINE_AA
    fontScale = fontScale * scale_factor
    
    thickness_o = int(thickness * scale_factor)
    thickness_i = int((thickness * 0.6) * scale_factor)

    # have an inner and outer radius for the circle to "emulate" an black outline
    radius_o = int(radius * scale_factor)
    radius_i = int((radius * 0.8) * scale_factor)

    for emg_name, emg_loc in scheme.locations.items():
        name = scheme.mapping.get(emg_name, emg_name)
        x, y = rel_to_abs(emg_loc[0], emg_loc[1], size=shape)

        canvas = cv2.circle(canvas, (x, y), radius=radius_o, color=( 0, 0, 0),   thickness=-1, lineType=lineType)
        canvas = cv2.circle(canvas, (x, y), radius=radius_i, color=color_circle, thickness=-1, lineType=lineType)

        text_size = cv2.getTextSize(name, fontFace=fontFace, fontScale=fontScale, thickness=thickness)[0]
        x -= text_size[0] // 2
        y += int(text_size[1]) + int(radius_o * 1.3)

        canvas = cv2.putText(canvas, name, (x, y), fontFace=fontFace, fontScale=fontScale, color=(  0,   0,   0), thickness=thickness_o, lineType=lineType)
        canvas = cv2.putText(canvas, name, (x, y), fontFace=fontFace, fontScale=fontScale, color=(255, 255, 255), thickness=thickness_i, lineType=lineType)

    if draw_outerhull:
        for x, y in face_model.get_outer(shape=shape):
            canvas = cv2.drawMarker(canvas, (x, y), markerType=cv2.MARKER_DIAMOND, color=(0, 0, 0), markerSize=radius_o, thickness=thickness_i)
    return canvas


def interpolate(
    scheme: Scheme,
    emg_values: dict[str, float],
    shape: tuple[int, int] = (512, 512),
    vmin: float = 0.0,
    vmax: Optional[float] = None,
    mirror: bool = False,
    mirror_plane_width: int = 2,
    missing_value: float = 0.0,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if not mirror:
        return __interpolate(scheme, emg_values, shape, vmin, vmax, missing_value)

    emg_values_mirrored_l, emg_values_mirrored_r = scheme.mirror(emg_values)

    if vmax is None:
        vmax = max(emg_values.values())

    interpolation_n = __interpolate(scheme, emg_values,            shape, vmin, vmax, missing_value)
    interpolation_l = __interpolate(scheme, emg_values_mirrored_l, shape, vmin, vmax, missing_value)
    interpolation_r = __interpolate(scheme, emg_values_mirrored_r, shape, vmin, vmax, missing_value)

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
    missing_value: float = 0.0,
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

    emg_values = scheme.validify(emg_values, missing_value=missing_value)
    canvas = np.zeros(shape, dtype=np.float32)
    outer_dict = {f"O{i}" : (x,y) for i, (x,y) in enumerate(face_model.get_outer(shape=shape))}
    
    keys_sorted_semg = sorted(scheme.locations.keys())
    keys_sorted_hull = sorted(outer_dict.keys())

    # get the values for each location
    v  = np.array([emg_values[k][0] for k in keys_sorted_semg] + [0] * len(keys_sorted_hull))
    xy = np.array([emg_values[k][1] for k in keys_sorted_semg] + [outer_dict[k] for k in keys_sorted_hull])

    vmin = vmin or v.min()
    vmax = vmax or v.max()
    lmax = v.max()

    # prepare the data for RBF interpolation
    p = xy.reshape(-1, 2)
    v =  v.reshape(-1, 1)
    x_grid = np.mgrid[-100 : 100 : canvas.shape[0] * 1j, -100 : 100 : canvas.shape[1] * 1j].reshape(2, -1).T
    Z = interp.RBFInterpolator(p, v, kernel="thin_plate_spline", smoothing=0.0)(x_grid)
    # reshape the data to the correct shape, and transpose it such it is rotated 90 degrees counter-clockwise
    Z = np.rot90(Z.reshape(canvas.shape[0], canvas.shape[1]))
    # all values smaller than 0 are set to 0
    Z[Z < vmin] = vmin

    # check if Z.max - vmin is really close 0, if so, set Z to 0
    if np.isclose(Z.max() - vmin, 0):
        Z = np.zeros_like(Z)
    else:
        Z = (Z - vmin) / (Z.max() - vmin)  # normalize the values to the range [0, 1]
    return Z * lmax


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
            colormap = matplotlib.colors.LinearSegmentedColormap.from_list("parula", parula_colormap)
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
        vmin = float(np.min(interpolation))
    if vmax is None:
        vmax = float(np.max(interpolation))

    if np.max(interpolation) > vmax or np.min(interpolation) < vmin:
        raise ValueError("electromyogram values are outside of the range [vmin, vmax]")

    # normalize the values between vmin and vmax
    if np.isclose(vmax - vmin, 0):
        interpolation = np.zeros_like(interpolation)
    else:
        interpolation = (interpolation - vmin) / (vmax - vmin)

    # scale the values to the range [0, 255]
    interpolation = (interpolation * 255).astype(np.uint8)
    colored = apply_colormap(interpolation, get_colormap(cmap))
    return colored

def postprocess(
    powermap: np.ndarray,
    remove_outer: bool = True,
    draw_triangle: bool = True,
    draw_locations: bool = False,
    triangles_alpha: float = 0.3,
    invert: bool = False,
    scheme: Optional[Scheme] = None,
    fill_value: int = 255,
) -> np.ndarray:
    powermap = powermap.copy()
    
    # scale the points to the current shape
    points = (face_model.points * powermap.shape[0]).astype(np.int32)
    thickness = math.ceil(powermap.shape[0] / 512) # thickness of the lines optimized for a 512x512 canvas
    
    if draw_triangle:
        color = 255 if powermap.ndim != 3 else (255, 255, 255)
        lines = np.zeros_like(powermap)
        lines = cv2.polylines(lines, [points[tri] for tri in face_model.triangles], isClosed=True, color=color, thickness=thickness)
        
        mask = np.zeros_like(powermap)
        mask[lines != 0] = 1

        if invert:
            lines = cv2.bitwise_not(lines)
            
        lines_masked = lines * mask
        power_masked = powermap * mask

        temp_blend = cv2.addWeighted(power_masked, 1-triangles_alpha, lines_masked, triangles_alpha, 0)
        powermap[mask == 1] = temp_blend[mask == 1]

    if remove_outer:
        hull = cv2.convexHull(points, returnPoints=True)
        mask = cv2.drawContours(np.zeros(powermap.shape[:2]), [hull], 0, 1, -1)
        powermap[mask == 0] = fill_value if powermap.ndim != 3 else [fill_value, fill_value, fill_value]
        
    if draw_locations:
        assert scheme is not None, "scheme must not be None if draw_locations is True"
        powermap = plot_locations(scheme=scheme, shape=powermap.shape[:2], do_postprocess=False, canvas=powermap)

    return powermap
