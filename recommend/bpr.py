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

        max_iter = (max_iter * self.recommender_data.preference_matrix.shape[0] *
                    self.recommender_data.preference_matrix.shape[1])
        u, s, vt = spMat.linalg.svds(self.recommender_data.preference_matrix, k)

        u = u * s

        user_matrix = u
        item_matrix = vt.T

        lambda_learning_rate = lambda_ * learning_rate

        for _ in xrange(max_iter):

            batch_user_idx = numpy.random.randint(user_matrix.shape[0])
            batch_item_idx_1 = numpy.random.randint(item_matrix.shape[0])
            batch_item_idx_2 = numpy.random.randint(item_matrix.shape[0])

            batch_user_vec = user_matrix[batch_user_idx]
            batch_item_vec_1 = item_matrix[batch_item_idx_1]
            batch_item_vec_2 = item_matrix[batch_item_idx_2]

            det_user_vec = item_matrix[batch_item_idx_1] - item_matrix[batch_item_idx_2]
            det_item_vec_1 = batch_user_vec
            det_item_vec_2 = -1 * batch_user_vec

            hat = numpy.inner(batch_user_vec, det_user_vec)

            user_matrix[batch_user_idx] += (learning_rate * (self._calc_bpr_comp(hat, det_user_vec))
                                            - lambda_learning_rate * batch_user_vec)

            item_matrix[batch_item_idx_1] += (learning_rate * (self._calc_bpr_comp(hat, det_item_vec_1))
                                            - lambda_learning_rate * batch_item_vec_1)


            item_matrix[batch_item_idx_2] += (learning_rate * (self._calc_bpr_comp(hat, det_item_vec_2))
                                            - lambda_learning_rate * batch_item_vec_2)



        self.user_matrix = user_matrix
        self.item_matrix = item_matrix.T

    @staticmethod
    def _calc_bpr_comp(x, delta_x):
        tmp = numpy.exp(-1 * x)

        tmp = tmp / (1 + tmp)

        return tmp * delta_x

