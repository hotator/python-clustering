#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
""" some helper functions for clustering """

import math
import numpy as np
from numpy.linalg import norm


def labelset_to_labels(labelset, n):
    """ converts [{1,2,3},{4,5,6}] to [0,0,0,1,1,1] """
    res = [0] * n
    for i, s in enumerate(labelset):
        for v in s:
            res[v] = i
    return res


def find_in_sublists(value, list_of_lists):
    """ search for value in list_of_lists """
    erg = [i for i, x in enumerate(list_of_lists) if value in x]
    return erg[0] if erg else -1


def get_other_node(value, edge):
    """ get the other node from an edge """
    assert value in edge  # asume value is in edge
    return edge[0] if edge[1] == value else edge[1]


def find_adjacent_edges(edge_index, edges):
    return [edge for edge in edges if edge_index in edge]


def get_related_points(point, points):
    res = []
    for p in points:
        if point in p:
            res += [p]
    return get_flat_list(res)


def get_flat_list(list_of_lists):
    """ get flat list from list of lists """
    return [item for sublist in list_of_lists for item in sublist]


def mydist(p1, p2):
    """ numpy version of distance between 2 points """
    return np.linalg.norm(np.array(p1)-np.array(p2))


def distance(p0, p1):
    """ non numpy version of distance between 2 points """
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)


def area(a, b, c):
    return 0.5 * norm(np.cross(b-a, c-a))


def beauty(vals):
    """ near 1: beauty, near 0: ugly """
    return min(vals) / max(vals) * 1.0


def get_colors():
    """ get the tableau20 colors for plots """
    # These are the "Tableau 20" colors as RGB.
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

    #res = ['#%02x%02x%02x' % triple for triple in tableau20]
    res = ["#{:02X}{:02X}{:02X}".format(*triple) for triple in tableau20]
    return res
