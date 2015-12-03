#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" clustering with combination """

from .base.Cluster import Cluster
from .base.HelperFunctions import distance
from itertools import combinations


class Combi(Cluster):
    def __init__(self, points):
        Cluster.__init__(self, points)

    def calculate(self):
        """ cluster algo """
        points = list(set(self.points))
        pair_list = sorted([(points.index(p1), points.index(p2), distance(p1, p2)) for p1, p2 in combinations(points, 2)], key=lambda x: x[2])
        loc = dict()

        for pair in pair_list:
            if pair[0] in loc:
                loc[pair[0]] += [pair[1]]
            else:
                loc[pair[0]] = [pair[1]]
            if pair[1] in loc:
                loc[pair[1]] += [pair[0]]
            else:
                loc[pair[1]] = [pair[0]]
            if len(loc) == len(points):
                break

        pattern = {key: 0 for key in loc}
        oldpattern = pattern.copy()
        run = 1

        while 0 in pattern.values():
            if run in pattern.values():
                while oldpattern != pattern:
                    oldpattern = pattern.copy()
                    for key, value in oldpattern.iteritems():
                        if value == run:
                            for origkey in loc[key]:
                                pattern[origkey] = run
                run += 1
            else:
                for key, value in pattern.iteritems():
                    if value == 0:
                        pattern[key] = run
                        break

        new_dict = {}
        for key, value in pattern.iteritems():
            new_dict.setdefault(value, set()).add(key)
        for key, value in new_dict.iteritems():
            self.result += [list(value)]
        self.result = [[points[i] for i in l] for l in self.result]
