#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" clustering with delauny triangulation """

from .base.Cluster import Cluster
from .base.HelperFunctions import get_connected_points, mydist
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from itertools import combinations


class Tri(Cluster):
    def __init__(self, points):
        Cluster.__init__(self, points)
        tri = Delaunay(self.points)
        self.simplices = tri.simplices
        self.connected = []

    def get_connected(self):
        dist = []
        for simplice in self.simplices:
            for t in combinations(simplice, 2):
                dist += [(mydist(self.points[t[0]], self.points[t[1]]), t[0], t[1])]
        dist = list(set(dist))
        dist.sort()
        self.connected = [(p1, p2) for _, p1, p2 in dist]

    def calculate(self):
        # TODO: for areas simple extra function like this - this = proxy
        self.get_connected()

        temp_set = set()
        pre_res = []
        for p in self.connected:
            temp_set |= {p[0], p[1]}
            pre_res += [[tuple(self.points[p[0]]), tuple(self.points[p[1]])]]
            if len(temp_set) == len(self.points):
                break
        self.result = get_connected_points(pre_res)

    def plot_simplices(self):
        plt.triplot(self.points[:, 0], self.points[:, 1], self.simplices.copy())
