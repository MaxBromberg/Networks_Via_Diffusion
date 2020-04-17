import plotter
import numpy as np


def info_space(x, y):
    # z = pow((1-x), y)
    # z = pow((2*x-1), 2)*np.sqrt(y)
    # z = pow((1-x), 2)*np.sqrt(y)
    z = np.sqrt(1-x)*np.sqrt(y)
    return z


plotter.plot_3d(info_space, x_range=[0, 1], y_range=[0, 1], piecewise=False)

