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

    def get_matrix(self):
        m = np.zeros((len(self.points), len(self.points)))

        for t in self.edge_list:
            x, y = min(t), max(t)
            m[(x, y)] = 1

        return m

    def gen_labels(self):
        m = self.get_matrix()
        g = nx.from_numpy_matrix(m)
        self.labels = list(nx.connected_components(g))

    def phase1(self):
        print('phase1')
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

    def get_short_points(self, i):
        return [x[1] for x in self.short_edges_list if x[0] == i] #+ [x[0] for x in self.short_edges_list if x[1] == i]

    def get_other_points(self, i):
        return [x[1] for x in self.other_edges_list if x[0] == i] #+ [x[0] for x in self.other_edges_list if x[1] == i]

    def plot_edge_list(self):
        self.plot_points()
        for i, j in self.edge_list:
            x, y = zip(*[self.points[i], self.points[j]])
            plt.plot(x, y)
        plt.show()

    def phase2(self):
        print('phase2')
        self.gen_labels()

        for i in range(len(self.points)):
            # 1. phase
            short_list = self.get_short_points(i)
            if short_list:
                # FIXME: if equal -> NOT the first but that one with shortest edge!
                c_label = self.labels.index(max([x for x in self.labels for n in short_list if n in x], key=len))

                # 2. phase
                other_list = self.get_other_points(i)
                if other_list:
                    if [self.labels.index(s) for s in self.labels if other_list[0] in s][0] != c_label:
                        for v in other_list:
                            if (i, v) in self.edge_list:
                                self.edge_list.remove((i, v))
                            #elif (v, i) in self.edge_list:
                            #    self.edge_list.remove((v, i))

                # 3. phase
                for v in short_list:
                    if v in self.labels[c_label]:
                        self.edge_list += [(i, v)]

    def get_last_points(self, i):
        return [x[1] for x in self.edge_list if x[0] == i] #+ [x[0] for x in self.edge_list if x[1] == i]

    def phase3(self):
        print('phase3')
        for i in range(len(self.points)):
            last_list = self.get_last_points(i)
            for v in last_list:
                if self.local_mean_list[i] + self.mean_st_dev_val < mydist(self.points[i], self.points[v]):
                    self.edge_list.remove((i, v))

    def calculate(self):
        self.phase1()
        #self.plot_edge_list()
        self.phase2()
        #self.plot_edge_list()
        self.phase3()
        self.plot_edge_list()
