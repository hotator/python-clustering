#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" multiple clustering algorithms """

import numpy as np
from cluster import *


if __name__ == '__main__':

    # get some data
    #pts, _ = make_blobs(n_samples=300, n_features=2, cluster_std=0.5, centers=2, random_state=42)
    #pts2, _ = make_blobs(n_samples=100, n_features=2, cluster_std=2, centers=1, random_state=564)
    #pts = np.concatenate((pts, pts2), axis=0)

    # antenne, moons, duenn, achsen, points_bloed, sixfour, noise, circles, blobs
    pts = np.genfromtxt('data/moons.csv', delimiter=',')

    # pts = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2],
    #                   [2, 0], [2, 1], [2, 2]])

    #X = [1, 6, 8, 3, 2, 2, 6, 6, 7, 7, 8, 8]
    #Y = [5, 2, 1, 5, 4, 6, 1, 8, 3, 6, 3, 7]
    #pts = list(zip(X, Y))

    #pts = np.random.rand(30, 2)

    #for point in pts:
    #    print(point)
    #plt.plot(pts[:, 0], pts[:, 1], 'o')
    #plt.show()

    # --------------------------------------------------------
    # Tri

    test = Tri(pts)
    test.calculate()
    #test.verify_result()
    #test.plot_simplices()
    test.plot_res()

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
    """
    vor = Voronoi(pts)
    print("Points: {}\n vertices: {}\n ridge_points: {}\n ridge_vertices: {}\n regions: {}\n point_region: {}".format(
        vor.points, vor.vertices, vor.ridge_points, vor.ridge_vertices, vor.regions, vor.point_region
    ))
    voronoi_plot_2d(vor)
    """

    # --------------------------------------------------------
    # Onion
    """
    test = Onion(pts, neighbours=2)
    #test.plot_points()
    #test.plot_vertices()
    test.calculate()
    #test.plot_points()
    #test.plot_connected_points()
    test.plot_res()
    """
    # --------------------------------------------------------
