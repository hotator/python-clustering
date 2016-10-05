#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
""" reimplementation of autoclust algorithm """

from .base.Cluster import Cluster
from .base.HelperFunctions import mydist, find_adjacent_edges, find_in_sublists, get_other_node
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


class Autoclust(Cluster):
    def __init__(self, points):
        Cluster.__init__(self, points)
        self.tri = Delaunay(self.points)
        self.acd = []  # autoclustdata
        self.edges = set()

        self.mean_st_dev = 0.0

        # only for plot + test
        self.all_short_edges = []
        self.all_long_edges = []
        self.all_other_edges = []

        # calculate
        self.calculate()

    def pre(self):
        #print('pre')
        self.get_edges()
        local_devs = []
        for i in range(len(self.points)):
            local_edges = find_adjacent_edges(i, self.edges)
            total_length = 0
            for edge in local_edges:
                total_length += mydist(self.points[edge[0]], self.points[edge[1]])
            mean_lenght = float(total_length / len(local_edges))
            sum_of_squared_diffs = 0
            for edge in local_edges:
                sum_of_squared_diffs += (mydist(self.points[edge[0]], self.points[edge[1]]) - mean_lenght) ** 2
            variance = float(sum_of_squared_diffs / len(local_edges))
            st_dev = np.sqrt(variance)
            local_devs.append(st_dev)
            self.acd.append({'local_mean': mean_lenght, 'local_stdev': st_dev})
        total = np.sum(local_devs)
        self.mean_st_dev = float(total/len(local_devs))

        for i in range(len(self.points)):
            local_edges = find_adjacent_edges(i, self.edges)
            short_edges = set()
            long_edges = set()
            other_edges = set()
            for edge in local_edges:
                length = mydist(self.points[edge[0]], self.points[edge[1]])
                if length < self.acd[i]['local_mean'] - self.mean_st_dev:
                    short_edges.add(edge)
                    self.all_short_edges.append(edge)
                elif length > self.acd[i]['local_mean'] + self.mean_st_dev:
                    long_edges.add(edge)
                    self.all_long_edges.append(edge)
                else:
                    other_edges.add(edge)
                    self.all_other_edges.append(edge)

            self.acd[i].update({'long_edges': long_edges, 'short_edges': short_edges, 'other_edges': other_edges})

    def phase1(self):
        #print('phase1')
        # remove long edges and short edges
        for i in range(len(self.points)):
            short_edges = self.acd[i]['short_edges']
            long_edges = self.acd[i]['long_edges']
            self.edges -= short_edges
            self.edges -= long_edges

    def phase2(self):
        #print('phase2')
        connected_components = self.find_connected_components()
        for i in range(len(self.points)):
            short_edges = self.acd[i]['short_edges']
            if short_edges:
                shortly_connected_components = set()
                for edge in short_edges:
                    other = get_other_node(i, edge)
                    g = find_in_sublists(other, connected_components)
                    shortly_connected_components.add(g)
                cv = 0
                if len(shortly_connected_components) > 1:
                    max_size = 0  # len, index
                    for scc in shortly_connected_components:
                        if len(connected_components[scc]) > max_size:
                            max_size = len(connected_components[scc])
                            cv = scc
                else:
                    cv = list(shortly_connected_components)[0]

                for edge in short_edges:
                    other = get_other_node(i, edge)
                    g = find_in_sublists(other, connected_components)
                    if cv == g:
                        self.edges.add(edge)
            # end if short edges not empty

            g = find_in_sublists(i, connected_components)
            if len(connected_components[g]) == 1:
                shortly_connected_components = set()
                for edge in short_edges:
                    other = get_other_node(i, edge)
                    g = find_in_sublists(other, connected_components)
                    shortly_connected_components.add(g)
                if len(shortly_connected_components) == 1:
                    self.edges |= short_edges

    def phase3(self):
        #print('phase3')
        for i in range(len(self.points)):
            edges_within_two = set()
            adjacent_edges = find_adjacent_edges(i, self.edges)
            edges_within_two |= set(adjacent_edges)
            for edge in adjacent_edges:
                other = get_other_node(i, edge)
                adjacent_to_other = find_adjacent_edges(other, self.edges)
                for a_edge in adjacent_to_other:
                    edges_within_two.add(a_edge)

            total_length = 0
            for edge in edges_within_two:
                total_length += mydist(self.points[edge[0]], self.points[edge[1]])
            if edges_within_two:  # FIXME: BUG?!
                local_2_mean = float(total_length / len(edges_within_two))

                for edge in edges_within_two:
                    if mydist(self.points[edge[0]], self.points[edge[1]]) > local_2_mean + self.mean_st_dev:
                        self.edges.remove(edge)

    def find_neighbors(self, pindex):
        return self.tri.vertex_neighbor_vertices[1][self.tri.vertex_neighbor_vertices[0][pindex]:self.tri.vertex_neighbor_vertices[0][pindex+1]]

    def get_edges(self):
        for i in range(len(self.points)):
            neighbors = self.find_neighbors(i)
            for n in neighbors:
                self.edges.add((min(i, n), max(i, n)))

    def get_matrix(self):
        m = np.zeros((len(self.points), len(self.points)))
        for t in self.edges:
            x, y = t
            m[(x, y)] = 1
        return m

    def find_connected_components(self):
        m = self.get_matrix()
        g = nx.from_numpy_matrix(m)
        return list(nx.connected_components(g))

    def plot_edges(self):
        self.plot_points()
        for i, j in self.edges:
            x, y = zip(*[self.points[i], self.points[j]])
            plt.plot(x, y)
        plt.show()

    def gen_result_from_labels(self):
        self.result = []
        for i in range(len(self.labels)):
            self.result += [[tuple(self.points[comp]) for comp in list(self.labels[i])]]

    def calculate(self):
        #plt.triplot(self.points[:, 0], self.points[:, 1], self.tri.simplices.copy())
        #plt.show()

        self.pre()
        self.phase1()
        #self.labels = self.find_connected_components()
        #self.gen_result_from_labels()
        #self.show_res()
        #self.plot_edges()

        self.phase2()
        #self.labels = self.find_connected_components()
        #self.gen_result_from_labels()
        #self.show_res()
        #self.plot_edges()

        self.phase3()
        self.labels = self.find_connected_components()
        #self.gen_result_from_labels()
        #self.show_res()
        #self.plot_edges()
