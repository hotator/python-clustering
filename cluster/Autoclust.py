#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" reimplementation of autoclust algorithm """

from .base.Cluster import Cluster
from .base.HelperFunctions import mydist
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


class Autoclust(Cluster):
    def __init__(self, points):
        Cluster.__init__(self, points)
        self.tri = Delaunay(self.points)
        self.point_infos = []
        self.edge_list = []

        self.local_mean_list = []
        self.local_st_dev_list = []
        self.mean_st_dev_val = 0.0
        self.relative_st_dev_list = []
        self.short_edges_list = []
        self.long_edges_list = []
        self.other_edges_list = []

        self.calculate()

    def gen_point_infos(self):
        l = [set() for _ in range(len(self.points))]
        for simplice in self.tri.simplices:
            for s in simplice:
                l[s] |= set(simplice) - {s}
        self.point_infos = l

    def find_neighbors(self, pindex):
        return self.tri.vertex_neighbor_vertices[1][self.tri.vertex_neighbor_vertices[0][pindex]:self.tri.vertex_neighbor_vertices[0][pindex+1]]

    def gen_all_connected(self):
        for i, p in enumerate(self.points):
            neighbors = self.find_neighbors(i)
            self.edge_list += [(i, n) for n in neighbors]

    def local_mean(self):
        res = []
        for i, p_info in enumerate(self.point_infos):
            temp = 0.0
            for s in p_info:
                temp += mydist(self.points[i], self.points[s]) / len(p_info)
            res += [temp]
        self.local_mean_list = res

    def local_st_dev(self):
        res = []
        for i, p_info in enumerate(self.point_infos):
            temp = 0.0
            for s in p_info:
                temp += (self.local_mean_list[i] - mydist(self.points[i], self.points[s]))**2 / len(p_info)
            res += [np.sqrt(temp)]
        self.local_st_dev_list = res

    def mean_st_dev(self):
        self.mean_st_dev_val = sum(self.local_st_dev_list) / len(self.local_st_dev_list)

    def relative_st_dev(self):
        self.relative_st_dev_list = [lml / self.mean_st_dev_val for lml in self.local_mean_list]

    def edges(self):
        for i, p_info in enumerate(self.point_infos):
            for p in p_info:
                dist = mydist(self.points[i], self.points[p])
                if dist < (self.local_mean_list[i] - self.mean_st_dev_val):
                    self.short_edges_list += [(i, p)]
                elif dist > (self.local_mean_list[i] + self.mean_st_dev_val):
                    self.long_edges_list += [(i, p)]
                else:
                    self.other_edges_list += [(i, p)]

    def phase1(self):
        self.gen_point_infos()
        self.local_mean()
        self.local_st_dev()
        self.mean_st_dev()
        self.relative_st_dev()
        self.edges()

        # phase 1
        self.gen_all_connected()

        # remove long edges and short edges
        for t in self.long_edges_list + self.short_edges_list:
            if (t[0], t[1]) in self.edge_list:
                self.edge_list.remove((t[0], t[1]))
            #elif (t[1], t[0]) in self.edge_list:
            #    self.edge_list.remove((t[1], t[0]))

        self.plot_points()

        for i, j in self.edge_list:
            x, y = zip(*[self.points[i], self.points[j]])
            plt.plot(x, y)
        plt.show()

    def phase2(self):
        # get connected components from self.edge_list
        pass

    def phase3(self):
        pass

    def calculate(self):
        self.phase1()
        self.phase2()
        self.phase3()
