#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" clustering with delauny triangulation """

from .base.Cluster import Cluster
from scipy.spatial import Delaunay
import numpy as np
from numpy.linalg import norm


def area(a, b, c):
    return 0.5 * norm(np.cross(b-a, c-a))


class TriVolume(Cluster):
    def __init__(self, points):
        Cluster.__init__(self, points)
        self.tri = Delaunay(self.points)

    def calculate(self, threshold=None):
        pass
