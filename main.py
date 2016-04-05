#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" multiple clustering algorithms """

import numpy as np
from cluster import *
from sklearn import datasets
from cluster.triclust import cluster
from multiprocessing import Pool
import os


def get_random_blobs():
    # some random blobs with differend density
    pts, _ = datasets.make_blobs(n_samples=300, n_features=7, cluster_std=0.5, centers=2, random_state=42)
    #pts2, _ = datasets.make_blobs(n_samples=100, n_features=37, cluster_std=2, centers=1, random_state=564)
    #return np.concatenate((pts, pts2), axis=0)
    return pts


def get_iris_from_sklearn():
    # import iris from sklearn.datasets
    iris = datasets.load_iris()
    pts = iris.data[:, :2]  # we only take the first two features.
    y = iris.target
    return pts, y


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
    if name in ['antenne', 'aggregation', 'compound', 'moons', 'duenn', 'achsen', 'points_bloed', 'sixfour', 'noise', 'circles', 'blobs']:
        return get_2d_data(name)
    print('all is wrong')


def compare_labels(real_labels, computed_labels):
    res = dict()
    for i, s in enumerate(computed_labels):
        for val in list(s):
            tmp = (real_labels[val], i)
            if tmp not in res:
                res[tmp] = 1
            else:
                res[tmp] += 1
    return res
    #z_list = list(zip(*t_list))
    #error1 = len(z_list[0]) != len(set(z_list[0]))  # cut one cluster in 2 or more
    #error2 = len(z_list[1]) != len(set(z_list[1]))  # merge 2 or more different cluster
    #print(error1, error2)


def calc(center):
    for _ in range(250):
        for n_sample in range(100, 501, 100):
            for n_feature in range(2, 5):
                #for center in range(2, 10):
                pts, labels = datasets.make_blobs(n_samples=n_sample, n_features=n_feature, cluster_std=0.5, centers=4)
                tri = Tri(pts)
                tri_res = compare_labels(labels, tri.labels)
                auto = Autoclust(pts)
                auto_res = compare_labels(labels, auto.labels)
                res_dict = {'tri': tri_res, 'auto': auto_res}

                with open('S' + str(n_sample) + 'F' + str(n_feature) + 'C' + str(center), 'a') as f:
                    print(res_dict, file=f)


def single_calc():
    n_sample = 100
    n_feature = 5
    cluster_std = 0.5
    center = 9

    pts, labels = datasets.make_blobs(n_samples=n_sample, n_features=n_feature, cluster_std=cluster_std, centers=center)
    tri = Tri(pts)
    tri_res = compare_labels(labels, tri.labels)
    auto = Autoclust(pts)
    auto_res = compare_labels(labels, auto.labels)
    res_dict = {'tri': tri_res, 'auto': auto_res}

    with open('S' + str(n_sample) + 'F' + str(n_feature) + 'C' + str(center), 'a') as f:
        print(res_dict, file=f)


def load():
    with open('S100F3C8', 'r') as f:
        for l in f.readlines():
            print(l)
            x = eval(l)
            print(x['auto'].keys())
            print(type(x))


if __name__ == '__main__':
    os.chdir('results')
    p = Pool()
    p.map(calc, range(3, 10))
    #p.map(single_calc, range(200, 401, 100))
    #single_calc()

    # get some data

    # antenne, aggregation, moons, duenn, achsen, points_bloed, sixfour, noise, circles, blobs
    # vessel
    # random
    #pts = get_data('random')
    #print(pts.shape)

    #pts, correct_labels = get_random_blobs()
    #print(correct_labels)

    # --------------------------------------------------------
    # Tri

    #tri = Tri(pts)
    #print(tri.labels)
    #compare_labels(correct_labels, tri.labels)
    #tri.gen_result_from_labels()
    #tri.show_res()

    # --------------------------------------------------------
    # Triclust test
    #print(cluster(pts)[0])

    # --------------------------------------------------------
    # Autoclust
    #auto = Autoclust(pts)
    #compare_labels(correct_labels, auto.labels)
    #print(auto.labels)

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
