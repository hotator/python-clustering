#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" reimplementation of triclust algorithm """

from matplotlib.tri import Triangulation
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as pl
import matplotlib.cm as plcm
from sklearn.cluster import KMeans as KMeans
from tempfile import mkstemp

__author__ = 'cbraune'


lls = {}
mels = {}
stds = {}


def line_length(p, q, ps):
    key = str(min(p, q))+"-"+str(max(p, q))
    if key not in lls:
        lls[key] = np.linalg.norm(ps[p] - ps[q])
    return lls[key]


def mean_edge_length(p, tri, ps):
    if p not in mels:
        edges = tri.edges[np.where(tri.edges == p)[0]]
        ll = [line_length(e[0], e[1], ps) for e in edges]
        mels[p] = np.mean(ll)
    return mels[p]


def std_point(p, tri, ps):
    if p not in stds:
        edges = tri.edges[np.where(tri.edges == p)[0]]
        ll = [line_length(e[0], e[1], ps) for e in edges]
        stds[p] = np.std(ll)
    return stds[p]


def dm(p, tri, ps):
    return std_point(p, tri, ps)/mean_edge_length(p, tri, ps)


def pdm(p, tri, ps):
    edg_idx, pos = np.where(tri.edges == p)
    edges = tri.edges[edg_idx]
    s = 0.0
    for i, e in enumerate(edges):
        m_i = mean_edge_length(p, tri, ps)
        m_j = mean_edge_length(e[1-pos[i]], tri, ps)
        sd = 0
        if m_i > m_j:
            sd = m_i - m_j
            sd /= line_length(p, e[1-pos[i]], ps)
        s += sd
    return s


# TRICLUST implementation according to
# Effective clustering and boundary detection algorithm based on Delaunay triangulation
# Dongquan Liu, Gleb V. Nosovskiy, Olga Sourina
# http://ac.els-cdn.com/S0167865508000500/1-s2.0-S0167865508000500-main.pdf?_tid=06b4cefe-f25d-11e4-a076-00000aab0f02&acdnat=1430744388_7225e82e13d3657c8b14f6e311ee2c92
def cluster(points, a=None, b=None, c=None, k=None, th=None, s_min=200, s_max=10000):

    points = points[:, 0:2]  # only use the first two dimensions

    tri = Triangulation(points[:, 0], points[:, 1])

    n = len(points)

    if n < s_min:
        if a is None:
            a = n/2000
        if b is None:
            b = 1
        if c is None:
            c = 0.5
    elif n >= s_max:
        if a is None:
            a = 2
        if b is None:
            b = 0.5
        if c is None:
            c = 1
    else:
        if a is None:
            a = (1.9*n+s_max/10-2*s_min)/(s_max-s_min)
        if b is None:
            b = 1 - (n-s_min)/(2*(s_max-s_min))
        if c is None:
            c = (0.5*n+0.5*s_max-s_min)/(s_max-s_min)

    if a+b+c <= 0:
        return "At least one parameter should be larger than zero!"

    fs = np.array([a * mean_edge_length(p, tri, points) + b * dm(p, tri, points) + c * pdm(p, tri, points)
                   for p in range(n)])

    if k is None:
        k = np.ceil((np.log10(n) / np.log10(2))+1)
    hist, bins = np.histogram(fs, bins=k)
    zeros = np.where(hist == 0)[0]
    if len(zeros) == 0:
        r1 = np.max(fs)
    else:
        r1 = bins[np.min(zeros) + 1]

    r2 = np.percentile(fs, 97)
    rc = r1 if n <= 5000 else min(r1, r2)

    km = KMeans(n_clusters=2, init=np.array([[np.min(fs[fs <= rc])], [np.mean(fs[fs <= rc])]]), n_init=1)
    km.fit(np.reshape(fs[fs <= rc], (len(fs[fs <= rc]), 1)))

    if th is None or th is 'arithmetic':
        th = np.mean(km.cluster_centers_)
    else:
        th = sps.hmean(km.cluster_centers_)

    boundary = fs > th
    inner = ~boundary
    index = np.argmax(inner)

    points_added = np.zeros_like(inner, dtype=bool)
    points_added[index] = True
    queue = {index}  # set of indices...

    clusters = []
    loop = True
    counter = 0
    while loop:
        counter += 1
        c = set()
        while len(queue) > 0:
            for e in tri.edges:
                if e[0] == index:
                    if inner[e[1]] and not points_added[e[1]]:
                        queue.add(e[1])
                    c.add(e[1])
                    points_added[e[1]] = True
                if e[1] == index:
                    if inner[e[0]] and not points_added[e[0]]:
                        queue.add(e[0])
                    c.add(e[0])
                    points_added[e[0]] = True
            index = queue.pop()
        clusters.append(np.array(list(c)))
        index = np.argmax(np.logical_and(inner, ~points_added))
        queue = {index}
        points_added[index] = True
        if np.all(points_added) or counter > 100:
            loop = False

    noise = []
    non_noise = []
    for c in clusters:
        if np.all(boundary[c]):
            noise.append(c)
        else:
            non_noise.append(c)

    labels = -np.ones((n,), dtype=np.int)
    for i in range(n):
        if not np.any([i in j for j in noise]):
            labels[i] = np.argmax([i in j for j in non_noise])

    _, file_name_1 = mkstemp(suffix=".png", prefix='tmp_cla_tri_')
    _, file_name_2 = mkstemp(suffix=".png", prefix='tmp_cla_tri_')
    _, file_name_3 = mkstemp(suffix=".png", prefix='tmp_cla_tri_')

    fig = pl.figure()
    pl.title("Clustering result")
    pl.gca().set_aspect('equal')
    for l in np.unique(labels):
        if l == -1:
            pl.plot(points[labels == l, 0], points[labels == l, 1], 'k.')
        else:
            pl.plot(points[labels == l, 0], points[labels == l, 1], 'o')
    pl.savefig(file_name_1)
    pl.close(fig)

    fig = pl.figure()
    pl.title("Delaunay triangulation, gouraud-shaded according to TRICLUST")
    pl.gca().set_aspect('equal')
    pl.tripcolor(tri, fs, edgecolors='k', cmap=plcm.rainbow, shading='gouraud')
    pl.plot(points[:, 0], points[:, 1], 'o')
    pl.colorbar()
    pl.savefig(file_name_2)
    pl.close(fig)

    fig = pl.figure()
    pl.title("TRICLUST Histogram")
    pl.hist(fs, bins=k)
    pl.savefig(file_name_3)
    pl.close(fig)

    return labels, [file_name_1, file_name_2, file_name_3]
