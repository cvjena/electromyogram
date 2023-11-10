__all__ = ["Scheme", "Kuramoto", "Fridlund", "FACS", "Blendshapes"]

import abc
from typing import Optional

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
    mapping: dict[str, str] = {}

    locations: dict[str, tuple[float, float]] = None

    def __init__(self) -> None:
        if self.locations is None:
            raise ValueError("Locations have to be implemented by the sub classes.")
 
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
        "med Front re": "F1",
        "med Front li": "F2",
        "lat Front re": "F3",
        "lat Front li": "F4",
        "Corr re": "F5",
        "Corr li": "F6",
        "Deprsup re": "F7",
        "Deprsup li": "F8",
        "Llsup re": "F9",
        "Llsup li": "F10",
        "OrbOr re": "F11",
        "OrbOr li": "F12",
        "DAO re": "F13",
        "DAO li": "F14",
        "Ment re": "F15",
        "Ment li": "F16",
        "OrbOc re": "F17",
        "OrbOc li": "F18",
        "Zyg re": "F19",
        "Zyg li": "F20",
        "Mass re": "F21",
        "Mass li": "F22",

    }
    pairs_L = ["F14", "F12", "F16", "F22", "F20", "F10", "F18", "F4", "F2", "F6", "F8"]
    pairs_R = ["F13", "F11", "F15", "F21", "F19",  "F9", "F17", "F3", "F1", "F5", "F7"]

    # fmt: off
    locations = {
         "F1": [-10.0,  65.0],
         "F2": [ 10.0,  65.0],
         "F3": [-32.0,  60.0],
         "F4": [ 32.0,  60.0],
         "F5": [-22.0,  52.0],
         "F6": [ 22.0,  52.0],
         "F7": [ -9.0,  49.0],
         "F8": [  9.0,  49.0],
         "F9": [-28.0, -12.0],
        "F10": [ 28.0, -12.0],
        "F11": [-21.0, -52.0],
        "F12": [ 21.0, -52.0],
        "F13": [-26.0, -69.0],
        "F14": [ 26.0, -69.0],
        "F15": [ -6.0, -65.0],
        "F16": [  6.0, -65.0],
        "F17": [-42.0,  10.0],
        "F18": [ 42.0,  10.0],
        "F19": [-53.0, -23.0],
        "F20": [ 53.0, -23.0],
        "F21": [-70.0, -50.0],
        "F22": [ 70.0, -50.0],

    }
    # fmt: on
    
class FACS(Scheme):
    # which are used for example in OpenFace2 [1,2,4,5,6,7,9,10,12,14,15,17,20,23,25,26]
    locations = {
        "AU1_R": (-10.0, 65.0),
        "AU1_L": ( 10.0, 65.0),
        
        "AU2_R": (-32.0, 60.0),
        "AU2_L": ( 32.0, 60.0),
        
        "AU4_R": ( -9.0, 49.0),
        "AU4_L": (  9.0, 49.0),
        
        # "AU5_R": ( 0.0, 0.0),
        # "AU5_L": ( 0.0, 0.0),
        
        "AU6_R": (-42.0, 10.0),
        "AU6_L": ( 42.0, 10.0),
        
        # "AU7_R": (0.0, 0.0),
        # "AU7_L": (0.0, 0.0),
        
        "AU9_R": (-28.0, -12.0),
        "AU9_L": ( 28.0, -12.0),
        
        # "AU10_R": (-28.0, -12.0),
        # "AU10_L": ( 28.0, -12.0),
        
        "AU12_R": (-53.0, -23.0),
        "AU12_L": ( 53.0, -23.0),
        
        # TODO location need confirmation 
        "AU14_R": (-40.0, -35.0),
        "AU14_L": ( 40.0, -35.0),
        
        "AU15_R": (-26.0, -69.0),
        "AU15_L": ( 26.0, -69.0),
        
        "AU17_R": (-6.0, -65.0),
        "AU17_L": ( 6.0, -65.0),
        
        # TODO location need confirmation
        "AU20_R": (-53.0, -45.0),
        "AU20_L": ( 53.0, -45.0),
        
        "AU23_R": (-21.0, -52.0),
        "AU23_L": ( 21.0, -52.0),
        
        # "AU25_R": (0.0, 0.0),
        # "AU25_L": (0.0, 0.0),
        
        "AU26_R": (-70.0, -50.0),        
        "AU26_L": ( 70.0, -50.0),
    }

class Blendshapes(Scheme):
    # X  1 - browDownLeft
    # X  2 - browDownRight
    # X  3 - browInnerUp
    # X  4 - browOuterUpLeft
    # X  5 - browOuterUpRight
    # X  6 - cheekPuff
    # X  7 - cheekSquintLeft
    # X  8 - cheekSquintRight
    # X  9 - eyeBlinkLeft
    # X 10 - eyeBlinkRight
    # X 11 - eyeLookDownLeft
    # X 12 - eyeLookDownRight
    # - 13 - eyeLookInLeft
    # - 14 - eyeLookInRight
    # - 15 - eyeLookOutLeft
    # - 16 - eyeLookOutRight
    # X 17 - eyeLookUpRight
    # X 18 - eyeLookUpRight
    # X 19 - eyeSquintLeft
    # X 20 - eyeSquintRight
    # X 21 - eyeWideLeft
    # X 22 - eyeWideRight
    # X 23 - jawForward
    # - 24 - jawLeft
    # X 25 - jawOpen
    # - 26 - jawRight
    # X 27 - mouthClose
    # X 28 - mouthDimpleLeft
    # X 29 - mouthDimpleRight
    # X 30 - mouthFrownLeft
    # X 31 - mouthFrownRight
    # X 32 - mouthFunnel
    # X 33 - mouthLeft
    # - 34 - mouthLowerDownLeft
    # - 35 - mouthLowerDownRight
    # X 36 - mouthPressLeft
    # X 37 - mouthPressRight
    # X 38 - mouthPucker
    # X 39 - mouthRight
    # X 40 - mouthRollLower
    # X 41 - mouthRollUpper
    # X 42 - mouthShrugLower
    # X 43 - mouthShrugUpper
    # X 44 - mouthSmileLeft
    # X 45 - mouthSmileRight
    # X 46 - mouthStretchLeft
    # X 47 - mouthStretchRight
    # - 48 - mouthUpperUpLeft
    # - 49 - mouthUpperUpRight
    # X 50 - noseSneerLeft
    # X 51 - noseSneerRight
    # X 52 - tongueOut
    locations = {
        "browInnerUp": (0.0, 65.0),
        
        "browOuterUpRight": (-32.0, 60.0),
        "browOuterUpLeft":  ( 32.0, 60.0),
        
        "browDownRight": ( -9.0, 49.0),
        "browDownLeft":  (  9.0, 49.0),

        "cheekSquintRight": (-42.0, 10.0),
        "cheekSquintLeft":  ( 42.0, 10.0),
        
        "noseSneerRight": (-16.0, -0.0),
        "noseSneerLeft":  ( 16.0, -0.0),
        
        "mouthShrugUpper": (0.0, -25.0),
        
        "mouthRight": (-27.0, -38.0),
        "mouthLeft":  ( 27.0, -38.0),
        
        "mouthSmileRight": (-53.0, -8.0),
        "mouthSmileLeft":  ( 53.0, -8.0),
        
        "mouthDimpleRight": (-48.0, -24.0),
        "mouthDimpleLeft":  ( 48.0, -24.0),
        
        "mouthFrownRight": (-26.0, -69.0),
        "mouthFrownLeft":  ( 26.0, -69.0),
        
        "jawForward": (-0.0, -65.0),

        "mouthStretchRight": (-53.0, -42.0),
        "mouthStretchLeft":  ( 53.0, -42.0),

    }