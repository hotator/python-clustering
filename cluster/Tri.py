#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" clustering with delauny triangulation """

from .base.Cluster import Cluster
from .base.HelperFunctions import get_connected_points, mydist
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt


class Tri(Cluster):
    def __init__(self, points):
        Cluster.__init__(self, points)
        tri = Delaunay(self.points)
        self.simplices = tri.simplices

    def calculate(self):
        # TODO: for areas simple extra function like this - this = proxy
        dist = []
        for triple in self.simplices:
            dist += [(mydist(self.points[triple[0]], self.points[triple[1]]), triple[0], triple[1])]
            dist += [(mydist(self.points[triple[1]], self.points[triple[2]]), triple[1], triple[2])]
            dist += [(mydist(self.points[triple[2]], self.points[triple[0]]), triple[2], triple[0])]

        dist = list(set(dist))
        dist.sort()
        dist = [(p1, p2) for _, p1, p2 in dist]

        temp_set = set()
        pre_res = []
        for p in dist:
            temp_set |= {p[0], p[1]}
            pre_res += [[tuple(self.points[p[0]]), tuple(self.points[p[1]])]]
            if len(temp_set) == len(self.points):
                break
        self.result = get_connected_points(pre_res)

    def plot_simplices(self):
        plt.triplot(self.points[:, 0], self.points[:, 1], self.simplices.copy())
