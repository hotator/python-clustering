#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" multiple clustering algorithms """

import numpy as np
from cluster import *
from sklearn import datasets
from cluster.triclust import cluster
from multiprocessing import Pool
import matplotlib.pylab as pl
import re
import os
from sklearn.metrics.cluster import adjusted_rand_score
from timeit import default_timer as timer


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


def labelset_to_labels(labelset, n):
    res = [0] * n
    for i, s in enumerate(labelset):
        for v in s:
            res[v] = i
    return res


def calc(center):
    for _ in range(333):
        for n_sample in range(100, 501, 100):
            for n_feature in range(2, 5):
                #for center in range(2, 10):
                seed = np.random.randint(0, 10000)
                pts, labels = datasets.make_blobs(n_samples=n_sample, n_features=n_feature, cluster_std=0.5, centers=center, random_state=seed)
                labels = list(labels)
                tri = Tri(pts)
                #tri_res = compare_labels(labels, tri.labels)
                tri_labels = labelset_to_labels(tri.labels, n_sample)
                tri_res = adjusted_rand_score(labels, tri_labels)

                auto = Autoclust(pts)
                #auto_res = compare_labels(labels, auto.labels)
                auto_labels = labelset_to_labels(auto.labels, n_sample)
                auto_res = adjusted_rand_score(labels, auto_labels)

                res_dict = {'labels': labels, 'tri_label': tri_labels, 'tri_score': tri_res, 'auto_labels': auto_labels, 'auto_score': auto_res, 'seed': seed}

                with open('S' + str(n_sample) + 'F' + str(n_feature) + 'C' + str(center), 'a') as f:
                    print(res_dict, file=f)


def single_calc(n_sample):
    #n_sample = 1000
    n_feature = 2
    cluster_std = 0.5
    center = 2

    #for n_sample in [10, 50, 100, 500, 1000, 5000, 10000]:

    pts, labels = datasets.make_blobs(n_samples=n_sample, n_features=n_feature, cluster_std=cluster_std, centers=center)
    start = timer()
    tri = Tri(pts)
    end = timer()
    tri_time = end - start
    print(tri_time)

    #tri_res = compare_labels(labels, tri.labels)
    start = timer()
    auto = Autoclust(pts)
    end = timer()
    auto_time = end - start
    print(auto_time)
    #auto_res = compare_labels(labels, auto.labels)
    res_dict = {'tri': tri_time, 'auto': auto_time, 'samples': n_sample}

    with open('times', 'a') as f:
        print(res_dict, file=f)


def load():
    # regular expression
    orgfiles = os.listdir('results')
    test = re.compile("S500")
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
        with open('results/' + f_name, 'r') as f:
            for i, l in enumerate(f):
                x = eval(l)
                temp_tri += [x['tri_score']]
                temp_auto += [x['auto_score']]
        #res += [temp_tri, temp_auto]
        #labels += ['tri', 'auto']
        res_tri += [temp_tri]
        res_auto += [temp_auto]
        labels += [f_name[4:]]

    #print(res)

    # tri result
    pl.figure(figsize=(1366/96, 768/96), dpi=100)
    pl.title('tri ' + f_name[1:4] + ' points')
    pl.boxplot(res_tri, labels=labels)
    #pl.show()
    pl.savefig('tri'+f_name[1:4]+'.pdf')
    pl.close()

    # auto result
    pl.figure(figsize=(1366 / 96, 768 / 96), dpi=100)
    pl.title('auto ' + f_name[1:4] + ' points')
    pl.boxplot(res_auto, labels=labels)
    #pl.show()
    pl.savefig('auto'+f_name[1:4]+'.pdf')
    pl.close()


def plot_res(res_dict):
    res = []
    keyes = []

    for key in sorted(res_dict):
        res += [res_dict[key]]
        keyes += [key]

    res = list(zip(*res))
    ind = np.arange(len(res_dict))
    width = 0.2

    pl_tri = pl.bar(ind, res[0], width, color='b')
    pl_auto = pl.bar(ind + width, res[1], width, color='y')
    pl.legend((pl_tri, pl_auto), ("tri", "auto"))
    pl.xticks(ind + width, keyes)
    #pl.xlabel('')
    test = min(min(res[0]), min(res[1]))
    print(test)
    pl.ylim(test-0.05, 1.0)
    pl.ylabel('Correct rate (%)')
    pl.show()


def plot_bar(my_result, name="invalid", verb=False, save=False):
    """ plots a barplot for regular, second and surro """
    res = list(zip(*my_result))
    ind = np.array([0.25, 0.5, 0.75])
    width = 0.20
    pl0 = pl.bar(ind, res[0], width, color="b")
    pl1 = pl.bar(ind, res[1], width, color="y", bottom=res[0])
    align = np.add(res[0], res[1])
    pl2 = pl.bar(ind, res[2], width, color="c", bottom=align)
    align = np.add(align, res[2])
    pl3 = pl.bar(ind, res[3], width, color="r", bottom=align)
    pl.xticks(ind + width / 2., ["auto", "tri"])
    #my_len = sum(my_result[0])
    #pl.yticks(range(0, my_len+1, int(my_len/3)),
    #    [x/float(my_len) for x in range(0, my_len+1, int(my_len/3))])
    pl.ylabel("values")
    pl.xlabel("algorithm")
    pl.legend((pl0[0], pl1[0], pl2[0], pl3[0]), ("correct", "error1", "error2", "errorboth"))
    print(res)
    #if save:
    #    pl.savefig(name + ".pdf")
    pl.show()


def plot_times():
    res = []
    with open('times', 'r') as f:
        for l in f.readlines():
            x = eval(l)
            res += [(x['samples'], x['auto'], x['tri'])]

    res = list(zip(*res))
    pl_auto = pl.plot(res[0], res[1])
    pl_tri = pl.plot(res[0], res[2])
    pl.xscale('log')
    pl.ylabel("time (in seconds)")
    pl.xlabel("points (logarithmic scale)")
    pl.legend((pl_auto[0], pl_tri[0]), ('auto', 'tri'))

    pl.show()


def static_test():
    files = ['aggregation', 'compound', 'moons', 'circles']
    for f in files:
        data = np.genfromtxt('data/' + f + '.csv', delimiter=',')
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
                    'auto_labels': auto_labels, 'auto_score': auto_res, 'auto_time': auto_time, 'name': f}

        with open('res', 'a') as fi:
            print(res_dict, file=fi)


if __name__ == '__main__':
    #static_test()
    name = 'aggregation'
    data = np.genfromtxt('data/' + name + '.csv', delimiter=',')
    pts = data[:, :2]
    #labels = data[:, -1]

    #os.chdir('results')
    #load()
    #p = Pool()
    #p.map(calc, range(3, 10))
    #p.map(single_calc, range(200, 401, 100))
    #single_calc()
    #p.map(single_calc, [10, 50, 100, 500, 1000, 5000, 10000])
    #plot_times()

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

    tri = Tri(pts)
    #tri_labels = labelset_to_labels(tri.labels, len(labels))
    #print(tri.labels)
    #compare_labels(correct_labels, tri.labels)
    tri.gen_result_from_labels()
    tri.show_res()


    # --------------------------------------------------------
    # Autoclust
    '''
    auto = Autoclust(pts)
    auto_labels = labelset_to_labels(auto.labels, len(labels))
    print(adjusted_rand_score(labels, auto_labels))
    #compare_labels(correct_labels, auto.labels)
    #print(auto.labels)
    auto.gen_result_from_labels()
    auto.show_res()
    '''

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
