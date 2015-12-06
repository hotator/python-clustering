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
        self.tri = Delaunay(self.points)
        self.connected = []

    def find_neighbors(self, pindex):
        return self.tri.vertex_neighbor_vertices[1][self.tri.vertex_neighbor_vertices[0][pindex]:self.tri.vertex_neighbor_vertices[0][pindex+1]]

    def get_connected(self):
        dist = []
        for i, p in enumerate(self.points):
            neighbors = self.find_neighbors(i)
            dist += [(mydist(p, self.points[n]), i, n) for n in neighbors]
        print(dist)
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
        plt.triplot(self.points[:, 0], self.points[:, 1], self.tri.simplices.copy())
