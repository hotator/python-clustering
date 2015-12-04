#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" some helper functions for clustering """

import math
import numpy as np


def get_related_points(point, points):
    res = []
    for p in points:
        if point in p:
            res += [p]
    return get_flat_list(res)


def get_flat_list(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def list_magic(point, points):
    """
        :param point - (x,y)
        :param points - list of pointlists [[(1,2),(2,3)],[(3,4),(6,5)]]
        :return set of unique points
    """

    old = set()
    new = {point}

    while old != new:
        old = new
        new = set()
        for p in old:
            new |= set(get_related_points(p, points))

    return new


def get_connected_points_old(points):
    points_set = set(get_flat_list(points))
    res = []
    while points_set:
        temp_set = list_magic(points_set.pop(), points)
        res += [list(temp_set)]
        points_set = points_set - temp_set
    return res


def get_connected_points(points):
    zwi_erg = []
    end_erg = []
    temp_set = set()
    rest_list = points
    new_rest = []

    while True:
        if len(rest_list) == 0:
            end_erg += [zwi_erg + list(temp_set)]
            break
        if not temp_set:
            cur_point = rest_list[0][0]  # FIXME: could be dangerous!
            if zwi_erg:
                end_erg += [zwi_erg]
            zwi_erg = [cur_point]
        else:
            cur_point = temp_set.pop()
            zwi_erg += [cur_point]
        for var in rest_list:
            if cur_point in var:
                temp_set |= set(var) - {cur_point}
            else:
                new_rest += [var]
        rest_list = new_rest
        new_rest = []
    return end_erg


def mydist(p1, p2):
    """ calculate distance between 2 points (in 2D) using numpy """
    return np.linalg.norm(p1-p2)


def distance(p0, p1):
    """ calculate distance between 2 points (in 2D) """
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)
