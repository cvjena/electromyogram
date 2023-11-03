__all__ = [
    "interpolate",
    "plot_locations",
    "Kuramoto",
    "Fridlund",
    "Scheme",
    "colorize",
    "get_colormap",
    "annotate_locations",
    "postprocess"
]

from electromyogram.plot import (
    annotate_locations,
    colorize,
    get_colormap,
    interpolate,
    plot_locations,
    postprocess,
)

from electromyogram.schemes import Fridlund, Kuramoto, Scheme