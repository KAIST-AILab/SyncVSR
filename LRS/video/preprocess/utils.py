import numpy as np
from typing import Tuple
import pydub

def retrieve_txt(path):
    # replace .mp4 with .txt
    path = path.replace(".mp4", ".txt")
    # read text
    with open(path, "r") as f:
        text = f.read()
    return text

def pydub_to_np(audio: pydub.AudioSegment) -> Tuple[np.ndarray, int]:
    """
    https://stackoverflow.com/a/66922265/8380469
    Converts pydub audio segment into np.float32 of shape [channels ,duration_in_seconds*sample_rate],
    where each value is in range [-1.0, 1.0]. 
    Returns tuple (audio_np_array, sample_rate).
    """
    return np.array(audio.get_array_of_samples(), dtype=np.float32).reshape((audio.channels, -1)) / (
            1 << (8 * audio.sample_width - 1)), audio.frame_rate
