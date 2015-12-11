#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" multiple clustering algorithms """

import numpy as np
from cluster import *
#from sklearn.datasets import make_blobs
from sklearn import datasets

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from random import shuffle


def get_colors():
    # These are the "Tableau 20" colors as RGB.
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)

    shuffle(tableau20)  # FIXME: really?
    return tableau20


def plot_2d(data_set):
    """
        plot the results
        :param data_set:
            format: [[point, point, point], [point, point, point, point]...]
    """
    #colors = 'rgbcmyk'
    colors = get_colors()
    for i, point_list in enumerate(data_set):
        x, y = zip(*point_list)
        plt.scatter(x, y, color=colors[i % len(colors)], marker='o')
    print("Number of cluster: {}".format(len(data_set)))
    plt.show()


def plot_3d(data_set):
    """
        plot the results in 3d
        :param data_set:
            format: [[point, point, point], [point, point, point, point]...]
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #colors = 'rgbcmyk'
    colors = get_colors()
    for i, vals in enumerate(data_set):
        x, y, z = zip(*vals)
        ax.scatter(x, y, z, color=colors[i % len(colors)])
    print("Number of cluster: {}".format(len(data_set)))
    plt.show()


if __name__ == '__main__':

    # get some data

    # some random blobs with differend density
    #pts, _ = datasets.make_blobs(n_samples=300, n_features=2, cluster_std=0.5, centers=2, random_state=42)
    #pts2, _ = datasets.make_blobs(n_samples=100, n_features=2, cluster_std=2, centers=1, random_state=564)
    #pts = np.concatenate((pts, pts2), axis=0)

    # import iris from sklearn.datasets
    #iris = datasets.load_iris()
    #pts = iris.data[:, :2]  # we only take the first two features.
    #Y = iris.target

    # antenne, moons, duenn, achsen, points_bloed, sixfour, noise, circles, blobs
    pts = np.genfromtxt('data/moons.csv', delimiter=',')
    pts = np.array([tuple(x) for x in pts if x[-1] != 0.282297])
    pts = pts[:, :2]
    # add dimensions (pseudorandom)
    for _ in range(1):
        z = np.random.rand(len(pts), 1)
        pts = np.append(pts, z, 1)
    #print(pts.shape)

    """
    # manual choosen points
    X = [1, 6, 8, 3, 2, 2, 6, 6, 7, 7, 8, 8]
    Y = [5, 2, 1, 5, 4, 6, 1, 8, 3, 6, 3, 7]
    Z = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
    pts = np.asarray(list(zip(X, Y, Z)))
    print(pts)
    """
    #pts = np.random.rand(30, 2)

    #for point in pts:
    #    print(point)
    #plt.plot(pts[:, 0], pts[:, 1], 'o')
    #plt.show()

    # --------------------------------------------------------
    # Tri

    test = Tri(pts)
    test.calculate()
    #print(test.connected)
    #test.verify_result()
    #test.plot_simplices()
    #print(len(test.result))
    #plot_2d(test.result)
    plot_3d(test.result)

    # --------------------------------------------------------
    # Combi

    #test = Combi(pts)
    #test.calculate()
    #test.verify_result()
    #test.plot_simplices()
    #test.plot_res()

    # --------------------------------------------------------
    # Voro

    #test = Voro(pts)
    #test.calculate()

    #vor = Voronoi(pts)
    #print("Points: {}\n vertices: {}\n ridge_points: {}\n ridge_vertices: {}\n regions: {}\n point_region: {}".format(
    #    vor.points, vor.vertices, vor.ridge_points, vor.ridge_vertices, vor.regions, vor.point_region
    #))
    #voronoi_plot_2d(vor)

    # --------------------------------------------------------
    # Onion
    #test = Onion(pts, neighbours=2)
    #test.plot_points()
    #test.plot_vertices()
    #test.calculate()
    #test.plot_points()
    #test.plot_connected_points()
    #test.plot_res()

    # --------------------------------------------------------
