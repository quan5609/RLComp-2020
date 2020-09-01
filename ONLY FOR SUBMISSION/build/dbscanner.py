#!/usr/bin/env python3
import numpy as np
import math

UNCLASSIFIED = False
NOISE = None
EPS = 3


def _dist(p, q):
    return (np.absolute(p-q).sum())


def _eps_neighborhood(p, q, eps):
    # print("Points:", p, q, _dist(p, q))
    return _dist(p, q) < eps


def _region_query(m, point_id, eps):
    n_points = m.shape[1]
    seeds = []
    for i in range(0, n_points):
        if _eps_neighborhood(m[:, point_id], m[:, i], eps):
            seeds.append(i)
    return seeds


def _expand_cluster(m, classifications, point_id, cluster_id, eps, min_points):
    seeds = _region_query(m, point_id, eps)
    if len(seeds) < min_points:
        classifications[point_id] = NOISE
        return False
    else:
        classifications[point_id] = cluster_id
        for seed_id in seeds:
            classifications[seed_id] = cluster_id

        while len(seeds) > 0:
            current_point = seeds[0]
            results = _region_query(m, current_point, eps)
            if len(results) >= min_points:
                for i in range(0, len(results)):
                    result_point = results[i]
                    if classifications[result_point] == UNCLASSIFIED or \
                       classifications[result_point] == NOISE:
                        if classifications[result_point] == UNCLASSIFIED:
                            seeds.append(result_point)
                        classifications[result_point] = cluster_id
            seeds = seeds[1:]
        return True


def dbscan(m, eps, min_points):
    """Implementation of Density Based Spatial Clustering of Applications with Noise
    See https://en.wikipedia.org/wiki/DBSCAN

    scikit-learn probably has a better implementation

    Uses Euclidean Distance as the measure

    Inputs:
    m - A matrix whose columns are feature vectors
    eps - Maximum distance two points can be to be regionally related
    min_points - The minimum number of points to make a cluster

    Outputs:
    An array with either a cluster id number or dbscan.NOISE (None) for each
    column vector in m.
    """
    cluster_id = 1
    n_points = m.shape[1]
    classifications = [UNCLASSIFIED] * n_points
    for point_id in range(0, n_points):
        point = m[:, point_id]
        if classifications[point_id] == UNCLASSIFIED:
            if _expand_cluster(m, classifications, point_id, cluster_id, eps, min_points):
                cluster_id = cluster_id + 1
    return classifications


def gold_dbscan(golds):
    # golds = golds
    goldArrayX = [gold['posx'] for gold in golds]
    goldArrayY = [gold['posy'] for gold in golds]
    m = np.array([goldArrayX, goldArrayY])
    eps = EPS
    min_points = 1
    dbscanLabel = dbscan(m, eps, min_points)
    clusters = np.unique(dbscanLabel)
    clusterArray = []
    for cluster in clusters:
        indexes = np.where(np.array(dbscanLabel) == cluster)[0]
        goldArray = [golds[i] for i in indexes]
        clusterArray.append(Cluster(cluster, goldArray))
    return clusterArray
    # print(clusterArray[0]._id, clusterArray[0].goldArray)
    # assert dbscan(m, eps, min_points) == [1, 1, 1, 2, 2, 2, None]


class Cluster:
    def __init__(self, id, goldArray):
        self._id = id
        self.goldArray = goldArray
        self.total_gold = self.calculate_total_gold()
        self.center_x, self.center_y = self.calculate_center()

    def calculate_total_gold(self):
        totalGold = 0
        for gold in self.goldArray:
            totalGold += gold['amount']
        return totalGold

    def distanceToCluster(self, posx, posy):
        def mahattan(x1, y1, x2, y2):
            return abs(x1-x2) + abs(y1 - y2)
        distanceArray = list(map(lambda x: mahattan(
            x['posx'], x['posy'], posx, posy), self.goldArray))
        minDistance = min(distanceArray)

        return minDistance, self.goldArray[distanceArray.index(minDistance)]['posx'], self.goldArray[distanceArray.index(minDistance)]['posy']

    def calculate_center(self):
        if len(self.goldArray) == 0:
            return -1, -1
        center_x, center_y = 0, 0
        for gold in self.goldArray:
            center_x += gold["posx"]
            center_y += gold["posy"]
        return int(center_x/len(self.goldArray)), int(center_y/len(self.goldArray))

    def update(self):
        self.total_gold = self.calculate_total_gold()
        self.center_x, self.center_y = self.calculate_center()

    def checkEnermyInCluster(self, players):
        def mahattan(x1, y1, x2, y2):
            return abs(x1-x2) + abs(y1 - y2)

        countEnermy = 0
        for player in players:
            if mahattan(self.center_x, self.center_y, player["posx"], player["posy"]) <= EPS:
                countEnermy += 1
        return countEnermy


# if __name__ == "__main__":
#     gold_dbscan()
