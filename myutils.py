import re
import numpy as np


def pre_process_text(text):
    text = re.sub('[^a-zA-Z0-9 -]', '', str(text))
    text = " ".join(str(text).split())
    text = text.lower()

    return text


def pad_array(arr):
    # Find the length of the longest element in the array
    max_length = 300

    # Create an empty result array with the same shape as the input array
    result = np.empty_like(arr, dtype=object)

    # Iterate over each row in the array
    # Pad each element in the row to match the length of the longest element
    if len(arr) > max_length:
        arr = arr[:max_length]
    padded_arr = np.pad(arr, (0, max_length - len(arr)), mode='constant', constant_values=0)

    return padded_arr
