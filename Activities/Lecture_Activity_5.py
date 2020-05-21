import numpy as np


def vector_mean(a):
    """
    Takes input array and calculates the mean for each row. Supports any array dimensions.
    Generates a vector containing all of the calculated mean row values in vector form.
    :param a: Input array for which the function calculates the mean of each row.
    :return: Column vector containing the means of each row in a.
    """
    vector_mean_output = np.zeros(shape=(1, a.shape[0]))
    # Generates an array of zeros with one row and same number of columns as input array.
    for row in range(a.shape[0]):
        # Runs through every row in input array a.
        mu = np.mean(a[row])
        # set mu = the mean of the current row in input array a.
        vector_mean_output[0, row] = mu
        # sets the same column in the generated zero array to the calculated mean value.
    return vector_mean_output


randomized_array = np.random.randint(50, size=(5, 7))
# Generate random array with maximum value 50, with 5 rows and 7 columns.
output = vector_mean(randomized_array)
# Call vector_mean() function for randomly generated array.

print(randomized_array)
print(output)
# Print original array and its vector mean.
