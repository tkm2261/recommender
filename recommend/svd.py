#!/usr/bin/env python
# -*- coding:utf-8

from recommender import Recommender
from sklearn.neighbors import NearestNeighbors
import numpy
import pandas
import scipy.sparse as spMat

class SVD(Recommender):

    def fit(self, k=100, max_iter=15):
        if self.recommender_data.preference_matrix.shape[1] < k:
            k = self.recommender_data.preference_matrix.shape[1] - 1


        u, s, vt = spMat.linalg.svds(self.recommender_data.preference_matrix, k)

        u = u * s

        self.user_matrix = numpy.array([row/row.sum() for row in u])
        self.item_matrix = numpy.array([row/row.sum() for row in vt.T])

    def get_score(self, k=100, batch=10000):

        nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(self.item_matrix)

        scores = nbrs.kneighbors(self.user_matrix, return_distance=False)
        scores = [[self.recommender_data.map_idx2item[i] for i in row] for row in scores]

        return scores



