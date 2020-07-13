import sys
sys.path.append('../')
import plotter
import numpy as np


def info_space(x, y):
    """
     By design, x is the (scaled) proportion of connected node value to all node connections
     (scaled proportional to effective distance, thus the inverse relation via (1-x) while
     y is the edge's value (i.e. probability of transition, given normalization of A columns)
    """
    # z = pow((1-x), y)
    # z = pow((2*x-1), 2)*np.sqrt(y)
    # z = pow((1-x), 2)*np.sqrt(y)
    # z = np.sqrt(1-x)*np.sqrt(y)
    alpha = 0.9
    z = pow(x, (alpha - 1))*pow(y, alpha)
    # Now alpha does not tune between ine relative influences of eff_dist and edge value, but instead acts as a general 'responsiveness' notion.
    # Best tuned between 0 and 1, s.t. d^2 z/dy^2 < 0 (increased y leads to a decrease in the rate of y's increase)
    # z = pow(y / x, alpha)
    return z


plotter.plot_3d(info_space, x_range=[0.1, 1], y_range=[0, 1], piecewise=False, spacing=0.05)

