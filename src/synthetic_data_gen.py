import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


def plot_2plots(x, y1, y2):
    """
    Plot two plots on same figure.
    :param x: x range.
    :param y1: actual y data.
    :param y2: generated y data.
    :return: Nothing.
    """

    plt.plot(x, y1, color='blue')
    plt.scatter(x, y2, marker='*', color='red')
    plt.legend(['Actual', 'Generated'])
    plt.show()


def data_out(data: pd.DataFrame, filename: str = 'synthetic_data.csv'):
    """
    Outputs DataFrame to csv.
    :param data: data to output.
    :param filename: Filename to output.
    :return: None.
    """

    default_path = "/Users/jackhu/Downloads"
    data.to_csv(default_path + "/" + filename, index=False)


def data_generator(x_range: np.linspace,
                   y_actual,
                   random_weight=np.random.uniform(-0.1, 0.1, 100)):
    """
    Dataset generator for synthetic data.
    :param x_range: range of x values.
    :param y_actual: values of y actual data.
    :param random_weight: random weight for random data.
    :return: pandas Dataframe object.
    """

    y_out = y_actual

    # Filter out non-zero values.
    y_out[y_out <= 0.0] = np.nan

    # Generate a random array with the same shape as y_func_2
    random_array = np.random.rand(*y_out.shape)

    # Create a mask with a probability between 0 and 1.
    mask = random_array < 0.55

    # Set elements to np.nan based on the mask
    y_out[mask] = np.nan

    return pd.DataFrame({'x': x_range,
                         'y': y_out + random_weight})


if __name__ == "__main__":

    # Output data.
    num_data = 100
    x_data = np.linspace(0, 2, num_data)

    a, b, c, d, f, g = -6.5, -2, 5, 1.5, 1, 0.3
    y_func_2 = g + a * (x_data - f)**4 + b * (x_data - f)**3 + c * (x_data - f)**2 + d * (x_data - f)

    ds = data_generator(x_data.copy(), y_func_2.copy(), np.linspace(-0.2, 0.2, num_data))
    plot_2plots(x_data, y_func_2, ds.y)
