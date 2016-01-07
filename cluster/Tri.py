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
    def __init__(self, points, min_cluster_size=0, threshold=0):
        Cluster.__init__(self, points)
        self.tri = Delaunay(self.points)
        self.connected = []
        self.dists = []
        self.min_cluster_size = min_cluster_size
        self.threshold = threshold

        # call calculation
        self.calculate()

    def find_neighbors(self, pindex):
        return self.tri.vertex_neighbor_vertices[1][self.tri.vertex_neighbor_vertices[0][pindex]:self.tri.vertex_neighbor_vertices[0][pindex+1]]

    def gen_all_connected(self):
        for i, p in enumerate(self.points):
            neighbors = self.find_neighbors(i)
            self.dists += [(mydist(p, self.points[n]), i, n) for n in neighbors]

    def min_connect_generator(self):
        dist = list(set(self.dists))
        dist.sort()
        for _, p1, p2 in dist:
            yield (p1, p2)

    def gen_min_connected(self):
        temp_set = set()
        for p in self.min_connect_generator():
            self.connected += [p]
            temp_set |= {p[0], p[1]}
            if len(self.points) == len(temp_set):
                break

    def find_threshold(self):
        """ find threshold for gen_threshold_connected """
        # TODO: implement, maybe dbscan 1 dim
        pass

    def gen_threshold_connected(self):
        for d, p1, p2 in self.dists:
            if d < self.threshold:
                self.connected += [(p1, p2)]

    def get_matrix(self):
        m = np.zeros((len(self.points), len(self.points)))

        for t in self.connected:
            x, y = min(t), max(t)
            m[(x, y)] = 1

        return m

    def gen_labels(self):
        m = self.get_matrix()
        g = nx.from_numpy_matrix(m)
        self.labels = list(nx.connected_components(g))

    def gen_result_from_labels(self):
        self.result = []
        self.noise = []
        for i in range(len(self.labels)):
            if self.min_cluster_size and len(self.labels[i]) < self.min_cluster_size:
                self.noise += [tuple(self.points[comp]) for comp in list(self.labels[i])]
            else:
                self.result += [[tuple(self.points[comp]) for comp in list(self.labels[i])]]

    def calculate(self):
        # generate all connections
        self.gen_all_connected()
        # type of connection
        if self.threshold:
            self.gen_threshold_connected()
        else:
            self.gen_min_connected()
        # generate labels
        self.gen_labels()
        # generate result from labels
        self.gen_result_from_labels()
        # show results
        self.show_res()

    def plot_simplices(self):
        plt.triplot(self.points[:, 0], self.points[:, 1], self.tri.simplices.copy())
        plt.show()
