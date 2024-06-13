import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


def plot_2plots(x, y1, y2):
    """
    Plot two plots on same figure.
    :param x: x range.
    :param y1: actual y data.
    :param y2: actual x data.
    :return: Nothing.
    """

    plt.scatter(x, y1, marker='o', color='blue')
    plt.scatter(x, y2, marker='*', color='red')
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


if __name__ == "__main__":
    x_range = np.linspace(0, 1, 100)
    y_actual = np.exp(x_range)
    y_synthetic = y_actual + np.random.uniform(-0.1, 0.1, 100)
    # print(y_actual[:10] - y_synthetic[:10])

    # Plot the generated data.
    # plot_2plots(x_range, y_actual, y_synthetic)

    # Output data.
    df = pd.DataFrame({'x': x_range, 'y': y_synthetic})
    data_out(df)
