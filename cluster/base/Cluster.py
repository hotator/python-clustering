#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" multiple clustering algorithms """

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt


class Cluster(object):
    def __init__(self, points):
        """ constuctor """
        # TODO: has verify -> as init argument
        # Attributes
        if points.shape[1] == 2:
            self.points = points[:, [0, 1]]  # update for bla
            self.verify = []
        else:
            self.points = points[:, range(points.shape[1] - 1)]
            self.verify = points[:, points.shape[1] - 1]
        self.colors = 'rgbcmyk'
        self.result = []

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
        """ read a csv-file into numpy array """
        self.points = np.genfromtxt(filename, delimiter=',')

    @staticmethod
    def save_csv(output, filename="test.csv"):
        """ write numpy array into csv-file """
        np.savetxt(filename, output, fmt="%.2f,%.2f,%d")

    def calculate(self):
        """ make something exciting """
        pass

    def plot_points(self):
        plt.plot(self.points[:, 0], self.points[:, 1], 'o')

    @staticmethod
    def plot_marker(x, y):
        plt.plot(x, y, "or", color="red", ms=10.0)

    def plot_res(self):
        """ plot the results """
        """ format: [[point, point, point], [point, point, point, point]...] """
        for i, point_list in enumerate(self.result):
            for point in point_list:
                plt.plot(point[0], point[1], self.colors[i % 7] + "o")
        print("Number of cluster: {}".format(len(self.result)))
        plt.show()

    def verify_result(self):
        # TODO: update function
        print("Verification")
        res_len = len(self.result)
        ver_len = len(set(x for x in self.verify))
        print([int(x) for x in self.verify])
        print(self.result)
        if ver_len > 0 and res_len > 0:
            print(ver_len, res_len)
        elif ver_len > 0:
            print("result not computed")
        elif res_len > 0:
            print("no verification data found")
        else:
            print("verification and result not found")
