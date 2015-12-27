#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" some helper functions for clustering """

import math
import numpy as np
from numpy.linalg import norm
from random import shuffle


def get_related_points(point, points):
    res = []
    for p in points:
        if point in p:
            res += [p]
    return get_flat_list(res)


def get_flat_list(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def mydist(p1, p2):
    return np.linalg.norm(np.array(p1)-np.array(p2))


def distance(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)


def area(a, b, c):
    return 0.5 * norm(np.cross(b-a, c-a))


def beauty(vals):
    # near 1: beauty, near 0: ugly
    return min(vals) / max(vals) * 1.0


def get_colors():
    # These are the "Tableau 20" colors as RGB.
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)

    shuffle(tableau20)  # FIXME: really?
    return tableau20
