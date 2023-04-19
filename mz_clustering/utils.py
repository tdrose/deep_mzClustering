import numpy as np


# Define a function to make a symmetric matrix from a non-symmetric matrix
def make_symmetric(matrix):
    # Use the maximum of the matrix and its transpose to make it symmetric
    symmetric_matrix = np.maximum(matrix, matrix.T)

    # Return the resulting symmetric matrix
    return symmetric_matrix
