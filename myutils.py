import re
import numpy as np


def pre_process_text(text):
    text = re.sub('[^a-zA-Z0-9 -]', '', str(text))
    text = " ".join(str(text).split())
    text = text.lower()

    return text


def pad_array(arr):
    # Find the length of the longest element in the array
    max_length = max(len(row) for row in arr)

    # Create an empty result array with the same shape as the input array
    result = np.empty_like(arr, dtype=object)

    # Iterate over each row in the array
    for i, row in enumerate(arr):
        # Pad each element in the row to match the length of the longest element
        padded_row = np.pad(row, (0, max_length - len(row)), mode='constant', constant_values=0)
        result[i] = padded_row

    return result
