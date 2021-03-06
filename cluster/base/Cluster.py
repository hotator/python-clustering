#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
""" multiple clustering algorithms """

import numpy as np
import matplotlib.pyplot as plt
from .HelperFunctions import get_colors
from mpl_toolkits.mplot3d import Axes3D
from random import shuffle


class Cluster:
    def __init__(self, points):
        # Attributes
        self.points = points
        self.labels = []
        self.result = []
        self.noise = []

    def __str__(self):
        """ String representation """
        return str(self.points)

    @staticmethod
    def area(p):
        return 0.5 * abs(sum(x0 * y1 - x1 * y0
                             for ((x0, y0), (x1, y1)) in Cluster.segments(p)))

    @staticmethod
    def segments(p):
        return zip(p, p[1:] + [p[0]])

    def open_csv(self, filename="la.csv"):
        self.points = np.genfromtxt(filename, delimiter=',')

    @staticmethod
    def save_csv(output, filename="test.csv"):
        np.savetxt(filename, output, fmt="%.2f,%.2f,%d")

    def calculate(self):
        """ make something exciting """
        pass

    def plot_me(self):
        plt.plot(self.points[:, 0], self.points[:, 1], 'o')
        plt.show()

    @staticmethod
    def plot_marker(x, y):
        plt.plot(x, y, "or", color="red", ms=10.0)

    def show_res(self, comp_list=None, filename=None, shuffle_colors=False):
        """
            plot the results in 3d
            format: [[point, point, point], [point, point, point, point]...]
            :param comp_list - [1,3,4] - plot only 1 3 and 4 as result
            :param filename - name of saving file, otherwise show
        """

        # print result
        print("clusters: {}".format(len(self.result)))
        if self.noise:
            print("noisepts: {}".format(len(self.noise)))

        # Plot
        fig = plt.figure()
        #colors = 'rgbcmyk'
        colors = get_colors()
        if shuffle_colors:
            shuffle(colors)
        markers = ('o', '+', 'x', '*', 's', 'p', 'h', 'H', 'D', 'd', '<', '>')
        plt.axis('equal')

        dim = len(self.result[0][0]) if self.result else 0
        if dim == 2:
            # 2D
            for i, point_list in enumerate(self.result):
                if comp_list and i not in comp_list:
                    continue
                x, y = zip(*point_list)
                plt.scatter(x, y, c=colors[i % len(colors)], marker=markers[i % len(markers)])
            # print noise
            if self.noise:
                x, y = zip(*self.noise)
                plt.scatter(x, y, c='b', marker='o')
        elif dim == 3:
            # 3D
            ax = fig.add_subplot(111, projection='3d')
            for i, vals in enumerate(self.result):
                if comp_list and i not in comp_list:
                    continue
                x, y, z = zip(*vals)
                ax.scatter(x, y, z, c=colors[i % len(colors)])
            # print noise
            if self.noise:
                x, y, z = zip(*self.noise)
                ax.scatter(x, y, z, c='b', marker='o')

        if dim in [2, 3]:
            if filename:
                plt.savefig(filename)
            else:
                plt.show()
