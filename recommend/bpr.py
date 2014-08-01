#!/usr/bin/env python
# -*- coding:utf-8

from recommender import Recommender
import numpy
import pandas
import scipy.sparse as spMat

class BPR(Recommender):

    def fit(self, k=100,
            max_iter=15,
            lambda_=0.1,
            learning_rate=0.1
            ):
        if self.recommender_data.preference_matrix.shape[1] < k:
            k = self.recommender_data.preference_matrix.shape[1] - 1


        u, s, vt = spMat.linalg.svds(self.recommender_data.preference_matrix, k)

        u = u * s

        user_matrix = u
        item_matrix = vt.T

        if user_batch_size > user_matrix.shape[0]:
            user_batch_size = user_matrix.shape[0]

        if item_batch_size > item_matrix.shape[0]:
            item_batch_size = item_matrix.shape[0]


        for itr in xrange(max_iter):

            batch_user_idx = numpy.random.randint(user_matrix.shape[0])
            batch_item_idx_1 = numpy.random.randint(item_matrix.shape[0])
            batch_item_idx_2 = numpy.random.randint(item_matrix.shape[0])

            batch_user_matrix = user_matrix[batch_user_idx]
            batch_item_matrix = item_matrix[batch_item_idx]



    @staticmethod
    def calc_bpr_comp(x, delta_x):
        tmp = numpy.exp(-1 * x)

        tmp = tmp / (1 + tmp)

        return tmp * delta_x

