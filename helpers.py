#!/usr/bin/env python3
""" Helper functions. """

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def to_step_function(x,
                     y):
    """ Convert the plot data consisting of the lists x and y to a step
    function. Convert x and y to numpy arrays."""

    n = 0
    while n < (len(x)-1):
        if y[n] != y[n+1]:
            x.insert(n+1, x[n+1])
            y = np.insert(y, n+1, y[n])
            n += 2
        else:
            n += 1
    return x, y


def to_normal_function(x,
                       y):
    """ Convert the step function consisting of the lists x and y to a normal
    function without steps. """

    n = 0
    while n < (len(x)-1):
        del x[n+1]
        del y[n+1]
        n += 1
    return x, y


def read_data(file, limit=None):
    """ Read 2-dimensional data from the file in <path> and return it as two
    lists. It possible to set a <limit> to the first dimension. Reads the first
    dimension as an integer and the second as a float.
    """

    x = []
    y = []
    with open(file) as file:
        for line in file:
            data = line.split()
            if data[0] != '#':
                if limit is not None:
                    if int(data[0]) > limit:
                        break
                x.append(int(float(data[0])))
                y.append(float(data[1]))
    return x, y


def adjust_axes():
    """ Adjust thes axes limits of the current plot by adding a margin. The
    function is designed for a figsize of (16, 9). """
    # Adjust axes limits
    xlims = list(plt.gca().get_xlim())
    xlims[0] = xlims[0] - (xlims[1]-xlims[0])*0.01*9/16
    xlims[1] = xlims[1] + (xlims[1]-xlims[0])*0.01*9/16
    plt.xlim(xlims)
    ylims = list(plt.gca().get_ylim())
    ylims[0] = ylims[0] - (ylims[1]-ylims[0])*0.01
    ylims[1] = ylims[1] + (ylims[1]-ylims[0])*0.01
    plt.ylim(ylims)


def truncate_colormap(cmap, minval=0.0, maxval=0.92, n=100, valuemax=None, totalmax=None):
    """ Return a truncated colormap that only includes the colors between
    <minval> and <maxval>. """
    if valuemax is not None and totalmax is not None:
        maxval = maxval*valuemax/totalmax
    new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval), cmap(np.linspace(minval, maxval, n)))
    return new_cmap