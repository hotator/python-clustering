#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" clustering with voronoi cells """

from .base.Cluster import Cluster
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt


class Voro(Cluster):
    """ a Voronoi cluster approach """

    def __init__(self, points):
        Cluster.__init__(self, points)
        self.vor = Voronoi(self.points, qhull_options='Qbb Qc Qz')
        self.point_region = self.vor.point_region
        self.vertices = self.vor.vertices
        self.regions = self.vor.regions
        self.ridge_points = self.vor.ridge_points
        self.ridge_vertices = self.vor.ridge_vertices
        self.point_areas = None
        self.area_indices = None

    def calculate(self):
        """ calculate """
        self.get_all_areas()
        self.get_small_area_indices()
        print(self.area_indices)

        plt.plot(self.points[:, 0], self.points[:, 1], '+')
        plt.plot(self.points[self.area_indices, 0], self.points[self.area_indices, 1], 'o')
        plt.plot(self.vertices[:, 0], self.vertices[:, 1], '*')
        for simplex in self.ridge_vertices:
            plt.plot(self.vertices[simplex, 0], self.vertices[simplex, 1], 'k-')

        plt.show()

    def get_all_areas(self):
        self.point_areas = []
        for i, point in enumerate(self.points):
            print("Point: {}, Area: {}".format(point, self.get_point_area(i)))
            self.point_areas += [self.get_point_area(i)]

    def get_small_area_indices(self):
        l = sorted(self.point_areas)
        self.area_indices = [self.point_areas.index(x) for x in l[:]]  # get 100 smallest indices

    def get_vertices(self, point_index):
        region_index = self.point_region[point_index]
        res = []
        for z in self.regions[region_index]:
            #if z == -1:
            #    res += [self.points[point_index]]
            #else:
            res += [self.vertices[z]]
        return res

    def get_point_area(self, point_index):
        vert = self.get_vertices(point_index)
        return self.area(vert)

    def plot_cells_2d(self):
        voronoi_plot_2d(self.vor)
