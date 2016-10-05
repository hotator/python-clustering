#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
""" multiple clustering algorithms """

import re
import os
from timeit import default_timer as timer

from sklearn import datasets
from sklearn.metrics.cluster import adjusted_rand_score
#from cluster.triclust import cluster
#from multiprocessing import Pool
import matplotlib.pylab as pl
import numpy as np
from cluster import Tri, Autoclust
from cluster.base.HelperFunctions import labelset_to_labels


def get_random_blobs():
    """ genernate some random blobs with differend density """
    pts, _ = datasets.make_blobs(n_samples=300, n_features=7, cluster_std=0.5, centers=2, random_state=42)
    #pts2, _ = datasets.make_blobs(n_samples=100, n_features=37, cluster_std=2, centers=1, random_state=564)
    #return np.concatenate((pts, pts2), axis=0)
    return pts


def get_iris():
    """ import iris dataset from sklearn.datasets """
    iris = datasets.load_iris()
    return iris.data, iris.target


def get_digit():
    """ import digit dataset from sklearn.datasets """
    digit = datasets.load_digits()
    return digit.data, digit.target


def get_vessel():
    """ import vessel dataset """
    vessel = np.genfromtxt('data/vesseltest.csv', delimiter=',')
    vessel = np.array([tuple(x) for x in vessel if x[-1] != 0.282297])
    vessel = vessel[:, :3]
    return vessel


def add_random_dimensions(points, dim=1):
    """ add dimensions (pseudorandom) """
    for _ in range(dim):
        z = np.random.rand(len(points), 1)
        res = np.append(points, z, 1)
    return res


def get_manual_3d_data():
    """ get manual choosen dataset """
    x = [1, 6, 8, 3, 2, 2, 6, 6, 7, 7, 8, 8]
    y = [5, 2, 1, 5, 4, 6, 1, 8, 3, 6, 3, 7]
    z = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
    return np.asarray(list(zip(x, y, z)))


def calculate_performance(center):
    """ calculates the performance of autoclust and tri clust on random blobs """
    for _ in range(333):
        for n_sample in range(100, 501, 100):
            for n_feature in range(2, 5):
                #for center in range(2, 10):
                seed = np.random.randint(0, 10000)
                pts, labels = datasets.make_blobs(n_samples=n_sample, n_features=n_feature, cluster_std=0.5, centers=center, random_state=seed)
                labels = list(labels)
                tri = Tri(pts)
                tri_labels = labelset_to_labels(tri.labels, n_sample)
                tri_res = adjusted_rand_score(labels, tri_labels)

                auto = Autoclust(pts)
                auto_labels = labelset_to_labels(auto.labels, n_sample)
                auto_res = adjusted_rand_score(labels, auto_labels)

                res_dict = {'labels': labels, 'tri_label': tri_labels, 'tri_score': tri_res, 'auto_labels': auto_labels, 'auto_score': auto_res, 'seed': seed}

                with open('S' + str(n_sample) + 'F' + str(n_feature) + 'C' + str(center), 'a') as f:
                    print(res_dict, file=f)


def calculate_times(n_sample):
    """ calculates time consumption of autoclust and tri clust on random blobs """
    #n_sample = 1000
    n_feature = 2
    cluster_std = 0.5
    center = 2

    #for n_sample in [10, 50, 100, 500, 1000, 5000, 10000]:
    pts, _ = datasets.make_blobs(n_samples=n_sample, n_features=n_feature, cluster_std=cluster_std, centers=center)
    start = timer()
    Tri(pts)
    end = timer()
    tri_time = end - start
    print(tri_time)

    start = timer()
    Autoclust(pts)
    end = timer()
    auto_time = end - start
    print(auto_time)
    res_dict = {'tri': tri_time, 'auto': auto_time, 'samples': n_sample}

    with open('times', 'a') as f:
        print(res_dict, file=f)


def plot_performance():
    """ plot performance compared as boxplots """
    # regular expression
    path_prefix = "/home/stephan/Dropbox/wcci2016/paper/figures/"
    orgfiles = os.listdir('results')
    test = re.compile("S100")
    files = list(filter(test.search, orgfiles))
    #files = orgfiles
    files.sort()

    #files = ['S5C1', 'S5C08', 'S5C06', 'S20C1', 'S20C08', 'S20C06', 'S50C1', 'S50C08', 'S50C06', 'S80C1', 'S80C08', 'S80C06']
    #files = ['S30C06la']
    #files = ['S100F2C3']
    print("files:", files)

    #res = []
    res_tri = []
    res_auto = []
    labels = []
    for f_name in files:
        temp_tri = []
        temp_auto = []
        with open('results/' + f_name, 'r') as res_file:
            for line in res_file:
                x = eval(line)
                temp_tri += [x['tri_score']]
                temp_auto += [x['auto_score']]
        res_tri += [temp_tri]
        res_auto += [temp_auto]
        labels += [f_name[4:]]

    #print(res)

    # tri result
    pl.boxplot(res_tri, labels=labels)
    pl.xticks(rotation=90)
    #pl.show()
    pl.savefig(path_prefix + 'tri'+f_name[1:4]+'.pdf')
    pl.close()

    # auto result
    pl.boxplot(res_auto, labels=labels)
    pl.xticks(rotation=90)
    #pl.show()
    pl.savefig(path_prefix + 'auto'+f_name[1:4]+'.pdf')
    pl.close()


def plot_times():
    """ plot results of time tests """
    res = []
    with open('times', 'r') as f:
        for l in f.readlines():
            x = eval(l)
            res += [(x['samples'], x['auto'], x['tri'])]

    res = list(zip(*res))
    #pl_auto = pl.plot(res[0], res[1])
    #pl_tri = pl.plot(res[0], res[2])
    #all_data = np.array(res[1]) / np.array(res[2])
    all_data = [x / res[2][i] for i, x in enumerate(res[1])]
    pl.plot(res[0], all_data)
    #pl.yscale('log')
    pl.ylabel("time (in seconds)")
    pl.xlabel("points (logarithmic scale)")
    #pl.legend((pl_auto[0], pl_tri[0]), ('auto', 'tri'))

    pl.show()


def static_test():
    """ do some tests """
    files = ['aggregation', 'compound', 'moons', 'circles']
    for f_name in files:
        data = np.genfromtxt('data/' + f_name + '.csv', delimiter=',')
        pts = data[:, :2]
        labels = data[:, -1]
        labels = list(labels)

        # tri
        start = timer()
        tri = Tri(pts)
        end = timer()
        tri_time = end - start
        tri_labels = labelset_to_labels(tri.labels, len(labels))
        tri_res = adjusted_rand_score(labels, tri_labels)

        # auto
        start = timer()
        auto = Autoclust(pts)
        end = timer()
        auto_time = end - start
        auto_labels = labelset_to_labels(auto.labels, len(labels))
        auto_res = adjusted_rand_score(labels, auto_labels)

        res_dict = {'labels': labels, 'tri_label': tri_labels, 'tri_score': tri_res, 'tri_time': tri_time,
                    'auto_labels': auto_labels, 'auto_score': auto_res, 'auto_time': auto_time, 'name': f_name}

        with open('res', 'a') as fi:
            print(res_dict, file=fi)


if __name__ == '__main__':
    #plot_times()
    #static_test()

    # get some data
    # antenne, aggregation, moons, duenn, achsen,
    # points_bloed, sixfour, noise, circles, blobs
    # flame, pathbased
    #name = 'moons'
    #data = np.genfromtxt('data/' + name + '.csv', delimiter=',')

    # load special datasets
    #data, l = get_iris()
    #data, l = get_digit()

    # generate datasets
    #data, l = datasets.make_moons(n_samples=500, noise=0.06)
    data, l = datasets.make_circles(n_samples=1000, noise=0.06, factor=0.5)
    #data, l = datasets.make_s_curve(n_samples=500, noise=0.02) # 3 dim
    pts = data[:, :2]

    '''
    names = ['achsen', 'aggregation', 'blobs', 'circles', 'compound', 'D31', 'moons', 'noise', 'R15', 'spiral']
    for name in names:
        #name = 'spiral'
        data = np.genfromtxt('data/' + name + '.csv', delimiter=',')
        pts = data[:, :2]
        labels = data[:, -1]
        tri = Autoclust(pts)
        tri.gen_result_from_labels()
        tri.show_res(filename='/home/stephan/Dropbox/wcci2016/TriRes/Auto_' + name + '.pdf')
    '''

    #load()
    #p = Pool()
    #p.map(calc, range(3, 10))
    #p.map(single_calc, range(200, 401, 100))
    #single_calc()
    #p.map(single_calc, [10, 50, 100, 500, 1000, 5000, 10000])
    #plot_times()

    #pts, correct_labels = get_random_blobs()
    #print(correct_labels)

    # --------------------------------------------------------
    # Tri

    tri = Tri(pts)
    #tri.plot_simplices()
    #tri_labels = labelset_to_labels(tri.labels, len(labels))
    #print(tri.labels)
    #tri.labels = [1, 1, 1, 2, 2, 2, 1, 2, 3, 2, 1, 1, 1, 1, 1, 3, 3, 3, 1, 3, 2, 3, 1, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 3, 1, 4, 2, 1, 3, 2, 1, 2, 2, 2, 2, 1, 1, 2, 3, 3, 1, 3, 2, 2, 2, 3, 2, 2, 1, 2, 3, 2, 2, 1, 2, 1, 2, 2, 3, 3, 1, 1, 3, 1, 2, 2, 1, 1, 3, 1, 1, 2, 2, 1, 1, 3, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 3, 3, 3, 1, 3, 1, 1, 2, 2, 1, 3, 1, 3, 3, 1, 2, 2, 3, 2, 1, 1, 3, 1, 1, 2, 2, 2, 1, 2, 3, 2, 1, 3, 2, 3, 1, 1, 2, 2, 3, 3, 1, 1, 1, 3, 2, 2, 1, 3, 2, 3, 2, 2, 2, 3, 1, 2, 1, 2, 1, 2, 2, 1, 3, 1, 3, 1, 1, 1, 3, 3, 3, 2, 1, 3, 1, 3, 1, 2, 3, 3, 1, 2, 2, 1, 1, 2, 3, 3, 1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 2, 3, 1, 2, 2, 2, 1, 3, 3, 3, 3, 2, 3, 1, 3, 1, 1, 3, 2, 2, 2, 3, 3, 3, 1, 3, 1, 2, 2, 3, 3, 3, 1, 1, 3, 2, 2, 1, 3, 3, 2, 3, 2, 2, 1, 2, 1, 3, 3, 2, 2, 2, 3, 3, 2, 1, 1, 2, 3, 3, 1, 1, 1, 2, 3, 3, 3, 3, 2, 2, 3, 1, 1, 2, 2, 2, 2, 5, 2, 3, 2, 2, 1, 2, 1, 1, 2, 3, 1, 1, 1, 1, 6, 3, 2, 3, 3, 1, 1, 1, 2, 2, 2, 1, 2, 3, 3, 1, 3, 2, 1, 1, 2, 1, 1, 1, 1, 2, 3, 3, 3, 3, 2, 3, 2, 1, 3, 2, 1, 3, 1, 1, 1, 1, 2, 3, 3, 1, 2, 3, 2, 2, 2, 1, 3, 3, 3, 3, 2, 1, 2, 3, 2, 3, 1, 3, 1, 3, 2, 2, 3, 2, 2, 3, 2, 3, 2, 3, 2, 2, 1, 3, 3, 3, 3, 3, 2, 2, 1, 3, 2, 3, 1, 7, 1, 1, 3, 2, 2, 3, 1, 1, 2, 1, 3, 2, 3, 1, 8, 2, 3, 3, 3, 2, 3, 3, 1, 3, 1, 1, 1, 1, 1, 2, 2, 2, 1, 3, 2, 3, 1, 1, 3, 3, 2, 1, 3, 3, 1, 2, 2, 1, 1, 2, 2, 2, 1, 1, 1, 3, 9, 3, 1, 2, 10, 1, 1, 2, 3, 2, 1, 1, 2, 2, 2, 1, 1, 1, 3, 1, 3, 3, 1, 1, 2, 1, 3, 3, 1, 1, 1, 1, 3, 2, 2, 1, 3, 2, 3, 1, 2, 3, 1, 3, 2, 2, 1, 2, 1, 3, 2, 1, 1, 3, 2, 1, 3, 3, 2, 1, 1, 1, 2, 1, 3, 2, 3, 2, 1, 1, 3, 2, 1, 3, 1, 1, 3, 1, 3, 3, 1, 2, 2, 3, 3, 3, 3, 2, 3, 3, 2, 2, 3, 2, 1, 2, 2, 2, 3, 2, 2, 3, 3, 3, 2, 3, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 1, 2, 1, 2, 1, 3, 1, 2, 3, 1, 3, 1, 2, 1, 2, 11, 3, 1, 2, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 3, 1, 3, 3, 1, 12, 3, 2, 3, 2, 3, 1, 3, 2, 1, 3, 1, 3, 3, 2, 3, 3, 2, 1, 1, 3, 2, 1, 1, 2, 1, 1, 3, 1, 3, 1, 1, 2, 2, 3, 3, 3, 3, 1, 2, 1, 3, 1, 3, 3, 3, 1, 3, 3, 2, 2, 2, 2, 3, 3, 2, 3, 3, 2, 1, 2, 2, 1, 1, 3, 2, 2, 2, 2, 2, 2, 3, 3, 2, 3, 2, 1, 3, 2, 2, 3, 2, 2, 1, 3, 1, 2, 3, 2, 1, 2, 3, 1, 1, 2, 1, 1, 3, 1, 1, 1, 2, 1, 1, 2, 1, 3, 2, 2, 1, 1, 3, 1, 2, 2, 1, 3, 2, 3, 3, 1, 2, 3, 2, 3, 3, 13, 3, 1, 1, 3, 3, 2, 3, 2, 2, 3, 2, 1, 1, 3, 2, 3, 3, 3, 1, 3, 2, 2, 3, 1, 2, 2, 3, 1, 3, 1, 2, 3, 3, 1, 1, 3, 2, 1, 1, 2, 3, 3, 3, 2, 2, 1, 1, 1, 2, 1, 2, 1, 3, 1, 3, 3, 1, 3, 2, 2, 2, 3, 2, 3, 1, 3, 1, 2, 2, 3, 3, 1, 3, 3, 3, 1, 1, 3, 2, 3, 2, 3, 3, 1, 3, 1, 1, 3, 1, 1, 3, 2, 3, 1, 2, 2, 1, 2, 3, 2, 3, 3, 1, 1, 3, 3, 3, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 3, 1, 14, 3, 2, 2, 3, 3, 2, 3, 3, 3, 1, 1, 2, 3, 1, 2, 2, 3, 2, 3, 1, 3, 15, 1, 3, 1, 3, 2, 3, 2, 1, 1, 3, 1, 1, 2, 3, 3, 3, 3, 1, 2, 2, 2, 3, 2, 3, 3, 1, 3, 1, 1, 1, 2, 2, 3, 2, 1, 1, 3, 2, 16, 3, 1, 1, 2, 3, 1, 2, 1, 3, 2, 1, 2, 3, 3, 3, 1, 1, 2, 3, 3, 1, 1, 1, 3, 3, 2, 1, 2, 3, 3, 2, 1, 2, 3, 1, 1, 2, 3, 2, 1, 3, 2, 3, 2, 3, 1, 1, 1, 1, 2, 3, 3, 1, 1, 3, 1, 1, 2, 1, 2, 2, 1, 1, 2, 1, 3, 3, 1, 1, 3, 1, 2, 3, 3, 1, 1, 1, 2, 3, 3, 2, 1, 1, 3, 3, 1, 3, 3, 1, 3, 2, 3, 1, 1, 2, 1, 3, 3, 3, 2, 2, 3, 1, 1, 1, 3, 3, 1, 1, 2, 3, 2, 3, 1, 2, 1, 1, 3, 2, 3, 3, 3, 2, 1, 2, 3, 1, 1, 3, 1, 3, 3, 1, 2, 1, 1, 1, 3, 3, 1, 1, 2, 2, 2, 2, 1, 3, 2, 2, 1, 3, 3, 3, 3, 2, 2, 2, 3, 2, 2, 2, 1, 2, 1, 3, 3, 2, 3, 1, 3, 3, 1, 17, 3, 3, 2, 1, 3, 3, 2, 2, 1, 3, 2, 2, 3, 2, 1, 3, 1, 2, 1, 1, 2, 2, 2, 1, 3, 2, 4, 3, 1, 1, 18, 2, 2, 2, 1, 1, 3, 2, 1, 2, 3, 1, 19, 2, 3, 1, 3, 3, 3, 2, 1, 1, 2, 20, 3, 2, 1, 3, 2, 1, 3, 1, 2, 2, 2, 3, 1, 1, 2, 3, 1, 1, 1, 3, 2, 2, 2, 2, 3, 2, 2, 3, 1, 3, 2, 1, 1, 3, 2, 3, 1, 3, 3, 2, 3, 2, 2, 3, 1, 3, 3, 3, 2, 3, 2, 1, 1, 3, 1, 2, 3, 1, 2, 1, 3, 1, 2, 2, 3, 1, 2, 2, 11, 1, 1, 3, 3, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 2, 2, 1, 2, 1, 3, 3, 2, 2, 1, 1, 2, 3, 1, 3, 2, 3, 1, 2, 1, 2, 3, 2, 1, 2, 1, 1, 3, 3, 3, 2, 3, 3, 1, 2, 2, 2, 1, 1, 3, 3, 2, 3, 3, 21, 2, 3, 1, 2, 1, 3, 2, 3, 1, 1, 3, 2, 3, 1, 1, 2, 1, 1, 3, 3, 1, 3, 2, 1, 1, 3, 2, 2, 3, 3, 2, 2, 2, 1, 2, 1, 2, 3, 2, 1, 1, 2, 3, 3, 1, 3, 2, 3, 3, 2, 2, 3, 2, 2, 1, 1, 2, 2, 3, 1, 3, 1, 1, 3, 1, 1, 2, 3, 3, 1, 2, 3, 3, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 3, 2, 2, 2, 2, 3, 3, 1, 1, 1, 2, 2, 2, 1, 1, 2, 3, 2, 2, 3, 1, 3, 3, 3, 3, 2, 2, 3, 1, 2, 2, 3, 3, 2, 3, 3, 3, 2, 2, 3, 3, 2, 3, 2, 3, 3, 2, 22, 1, 3, 23, 3, 3, 3, 3, 1, 3, 3, 2, 2, 2, 2, 2, 1, 2, 2, 2, 3, 2, 1, 24, 1, 1, 3, 3, 3, 2, 2, 2, 2, 1, 3, 2, 3, 3, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 1, 1, 1, 2, 1, 2, 3, 2, 3, 1, 1, 3, 3, 1, 3, 2, 1, 3, 1, 1, 2, 2, 1, 1, 2, 2, 3, 1, 2, 2, 2, 1, 1, 1, 3, 1, 1, 3, 3, 3, 2, 2, 1, 2, 1, 2, 1, 1, 3, 1, 2, 3, 2, 3, 3, 1, 2, 2, 1, 3, 1, 3, 3]
    tri.gen_result_from_labels()
    tri.show_res(shuffle_colors=True)

    # --------------------------------------------------------
    # Autoclust

    #auto = Autoclust(pts)
    #auto_labels = labelset_to_labels(auto.labels, len(labels))
    #print(adjusted_rand_score(labels, auto_labels))
    #print(auto.labels)
    #auto.gen_result_from_labels()
    #auto.show_res()

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
