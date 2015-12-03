#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" simple and not powerfull clustering algorithm using convex hull """

from .base.Cluster import Cluster
from .base.HelperFunctions import get_connected_points
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt


class Onion(Cluster):
    def __init__(self, points, neighbours=2):
        Cluster.__init__(self, points)

        # Attributes
        self.verti_list = []
        self.cur_point = ()
        self.neighbours = neighbours
        self.point_dict = dict()

        # functions
        self.get_vertices()

    def get_vertices(self):
        points_copy = np.copy(self.points)
        while points_copy.any():
            if len(points_copy) < 3:
                self.verti_list += [[(point[0], point[1]) for point in points_copy]]
                break
            hull = ConvexHull(points_copy)
            self.verti_list += [[(points_copy[x, 0], points_copy[x, 1]) for x in hull.vertices]]
            points_copy = np.delete(points_copy, list(hull.vertices), 0)

    def get_level(self):
        for i, p in enumerate(self.verti_list):
            if self.cur_point in p:
                return i
        return len(self.verti_list)-1

    def get_level_up_points(self):
        i = self.get_level()
        if i+1 >= len(self.verti_list):
            return []
        else:
            return self.verti_list[i+1]

    def get_level_down_points(self):
        i = self.get_level()
        if i == 0:
            return []
        else:
            return self.verti_list[i-1]

    def calc_nearest(self, n_points):
        res = [(np.linalg.norm(np.array(self.cur_point)-np.array(p)), p) for p in n_points]
        res.sort()
        return [erg[1] for erg in res]

    def get_hull_points(self):
        level = self.get_level()
        hull = self.verti_list[level]
        if len(hull) < 3:
            return []
        if hull[0] == self.cur_point:
            return [hull[1], hull[-1]]
        elif hull[-1] == self.cur_point:
            return [hull[0], hull[-2]]
        else:
            i = hull.index(self.cur_point)
            return [hull[i-1], hull[i+1]]

    def get_nearest_n(self, point, n=2):
        self.cur_point = tuple(point)
        level_up_points = self.get_level_up_points()
        level_down_points = self.get_level_down_points()
        # hull_points = self.get_hull_points()
        n_points = level_up_points + level_down_points
        res = self.calc_nearest(n_points)
        return res[:n]

    def get_all_neighbours(self):
        self.point_dict = {tuple(p): self.get_nearest_n(p, n=self.neighbours) for p in self.points}

    def calculate(self):
        self.get_all_neighbours()
        la = []
        for point in self.points:
            point = tuple(point)
            if point in self.point_dict:
                la += [[point] + self.point_dict[point]]
            else:
                la += [[point]]
        self.result = get_connected_points(la)

    def plot_connected_points(self):
        assert self.point_dict
        for point in self.points:
            for val in self.point_dict[tuple(point)]:
                cur = [tuple(point)]
                cur += [val]
                x, y = zip(*cur)
                plt.plot(x, y)
        plt.show()

    def plot_vertices(self):
        # for val in [-3]:
        for val in range(len(self.verti_list)):
            hull = self.verti_list[val]
            x, y = zip(*hull)
            # connect first and last elem
            x += (x[0],)
            y += (y[0],)
            plt.plot(x, y)
        plt.show()
