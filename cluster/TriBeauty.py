#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" clustering with delauny triangulation """

from .base.Cluster import Cluster
from scipy.spatial import Delaunay
from .base.HelperFunctions import beauty


class TriBeauty(Cluster):
    def __init__(self, points):
        Cluster.__init__(self, points)
        self.tri = Delaunay(self.points)
        self.beauty_list = []

    def get_beauty_vals(self):
        for val in self.tri.simplices:
            self.beauty_list += [(beauty(val), val)]

    def calculate(self, threshold=None):
        self.get_beauty_vals()
        print(self.beauty_list)
