#!/usr/bin/env python
# -*- coding:utf-8

from recommender import Recommender
import numpy
import pandas
import scipy.sparse as spMat

class SVD(Recommender):

    def fit(self, k=100, max_iter=15):
        if self.recommender_data.preference_matrix.shape[1] < k:
            k = self.recommender_data.preference_matrix.shape[1] - 1


        u, s, vt = spMat.linalg.svds(self.recommender_data.preference_matrix, k)

        u = u * s

        self.user_matrix = u
        self.item_matrix = vt




