#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" clustering with delauny triangulation """

from .base.Cluster import Cluster
from .base.HelperFunctions import mydist
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


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

    def get_minimum_connected(self):
        temp_set = set()
        i = len(self.connected)
        for i in range(len(self.connected)):
            temp_set |= {self.connected[i][0], self.connected[i][1]}
            if len(temp_set) == len(self.points):
                break
        self.connected = self.connected[:i+1]

    def get_matrix(self):
        m = np.zeros((len(self.points), len(self.points)))

        for t in self.connected:
            x, y = min(t), max(t)
            m[(x, y)] = 1

        return m

    def calculate(self):
        # TODO: for areas simple extra function like this - this = proxy
        self.get_connected()
        self.get_minimum_connected()
        m = self.get_matrix()

        G = nx.from_numpy_matrix(m)
        con_comp = list(nx.connected_components(G))
        for i in range(len(con_comp)):
            self.result += [[tuple(self.points[comp]) for comp in list(con_comp[i])]]

    def plot_simplices(self):
        plt.triplot(self.points[:, 0], self.points[:, 1], self.tri.simplices.copy())
