import numpy as np

def sort_with_indices(arr):
    sorted_indices = np.argsort(arr)
    sorted_array = np.sort(arr)
    return sorted_array, sorted_indices.tolist()

# Example usage
input_array = np.array([1, 7, 8, 5], dtype=float)
sorted_array, original_indices = sort_with_indices(input_array)

print(f"Input: {[1, 7, 8, 5]}")
print(f"Sorted array: {sorted_array}")
print(f"Original indices: {original_indices}")