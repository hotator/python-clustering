#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" multiple clustering algorithms """

import numpy as np
from cluster import *
from sklearn import datasets


def get_random_blobs():
    # some random blobs with differend density
    pts, _ = datasets.make_blobs(n_samples=300, n_features=3, cluster_std=0.5, centers=2, random_state=42)
    pts2, _ = datasets.make_blobs(n_samples=100, n_features=3, cluster_std=2, centers=1, random_state=564)
    return np.concatenate((pts, pts2), axis=0)


def get_iris_from_sklearn():
    # import iris from sklearn.datasets
    iris = datasets.load_iris()
    pts = iris.data[:, :2]  # we only take the first two features.
    Y = iris.target
    return pts, Y


def get_vessel():
    pts = np.genfromtxt('data/vesseltest.csv', delimiter=',')
    pts = np.array([tuple(x) for x in pts if x[-1] != 0.282297])
    pts = pts[:, :3]
    return pts


def get_2d_data(name=''):
    pts = np.genfromtxt('data/' + name + '.csv', delimiter=',')
    pts = pts[:, :2]
    return pts


def add_random_dimensions(pts, dim=1):
    # add dimensions (pseudorandom)
    for _ in range(dim):
        z = np.random.rand(len(pts), 1)
        pts = np.append(pts, z, 1)
    return pts


def get_manual_3d_data():
    # manual choosen points
    x = [1, 6, 8, 3, 2, 2, 6, 6, 7, 7, 8, 8]
    y = [5, 2, 1, 5, 4, 6, 1, 8, 3, 6, 3, 7]
    z = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
    return np.asarray(list(zip(x, y, z)))


def get_data(name=''):
    if name == 'vessel':
        return get_vessel()
    if name == 'random':
        return get_random_blobs()
    if name in ['antenne', 'moons', 'duenn', 'achsen', 'points_bloed', 'sixfour', 'noise', 'circles', 'blobs']:
        return get_2d_data(name)
    print('all is wrong')

if __name__ == '__main__':

    # get some data

    # antenne, moons, duenn, achsen, points_bloed, sixfour, noise, circles, blobs
    # vessel
    # random
    #pts = get_data('vessel')
    pts, bla = datasets.make_blobs(n_samples=2000, centers=8, n_features=3, cluster_std=0.5, shuffle=False)
    print(pts.shape)

    # --------------------------------------------------------
    # Tri

    test = Tri(pts)
    test.calculate()
    #test.plot_simplices()
    # TODO: implement accuracy function
    '''
    res_labels = np.zeros(len(pts), dtype=int)
    for i, l_list in enumerate(test.labels):
        for l in l_list:
            res_labels[l] = i
    #print(res_labels)

    quali = 0
    for i in range(len(bla)):
        if bla[i] == res_labels[i]:
            quali += 1
    print(quali/len(bla) * 1.)
    '''
    test.show_res()
    #test.plot_points()
    #test.plot_simplices()

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
