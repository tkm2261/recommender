#!/usr/bin/env python
# -*- coding:utf-8
import unittest
import os
import numpy
from recommend.dao import read_csv
from recommend.recommender import Recommender
from recommend.nmf import NMF
from recommend.svd import SVD
from recommend.lda import LDA
from recommend.bpr import BPR

from recommend.io import RecommenderData
PWD = os.getcwd()



class TestRecommend(unittest.TestCase):
    """
    def test_read_data(self):
        data = read_csv(PWD+"/data/test_data.csv").preference_matrix

        test_data = numpy.zeros((5, 6))
        test_data[0, 0] = 1
        test_data[1, 1] = 2
        test_data[2, 2] = 3
        test_data[3, 3] = 3
        test_data[0, 4] = 2
        test_data[4, 5] = 1

        self.assertTrue(numpy.allclose(test_data, data.todense()))

    def test_recommender(self):

        map_tmp_user = {i:i for i in xrange(5)}
        map_tmp_item = {i:i for i in xrange(3)}
        rd = RecommenderData(preference_matrix=None,
                        map_idx2user=map_tmp_user,
                        map_idx2item=map_tmp_item,
                        map_user2idx=map_tmp_user,
                        map_item2idx=map_tmp_item)

        rec = Recommender(rd)

        rec.user_matrix = numpy.array([[ 0.4218758 ,  0.69421827],
                                       [ 0.8040729 ,  0.67054794],
                                       [ 0.46242548,  0.17621366],
                                       [ 0.37208287,  0.1851422 ],
                                       [ 0.23163381,  0.82274935]])

        rec.item_matrix = numpy.array([[ 0.86450211,  0.27870829,  0.88082552],
                                       [ 0.47062048,  0.85398355,  0.47154423]])

        ans_matrix = numpy.array([[ 0.69142586,  0.71043127,  0.69895359],
                                   [ 1.01069631,  0.79673869,  1.02444094],
                                   [ 0.48269756,  0.27936538,  0.4904087 ],
                                   [ 0.40879814,  0.26181097,  0.41504283],
                                   [ 0.58745062,  0.76717268,  0.59199168]])

        test_index = [[1, 2, 0], [2, 0, 1], [2, 0, 1]]
        ans_index =  rec.predict([0, 1, 2], ranking=10, index=True)

        self.assertEqual(test_index, ans_index)

        ans_index =  rec.get_score()
        test_index = [[1, 2, 0], [2, 0, 1], [2, 0, 1], [2, 0, 1], [1, 2, 0]]
        self.assertEqual(test_index, ans_index)

    def test_nmf(self):

        map_tmp_user = {i:i for i in xrange(5)}
        map_tmp_item = {i:i for i in xrange(3)}
        preference_matrix = numpy.array([[ 0.69142586,  0.71043127,  0.69895359],
                                           [ 1.01069631,  0.79673869,  1.02444094],
                                           [ 0.48269756,  0.27936538,  0.4904087 ],
                                           [ 0.40879814,  0.26181097,  0.41504283],
                                           [ 0.58745062,  0.76717268,  0.59199168]])
        rd = RecommenderData(preference_matrix=preference_matrix,
                        map_idx2user=map_tmp_user,
                        map_idx2item=map_tmp_item,
                        map_user2idx=map_tmp_user,
                        map_item2idx=map_tmp_item)



        model = NMF(rd)
        model.fit(max_iter=2)
        test_index = [[1, 2, 0], [2, 0, 1], [2, 0, 1], [2, 0, 1], [1, 2, 0]]
        ans_index =  model.get_score()
        self.assertEqual(test_index, ans_index)

    def test_svd(self):

        map_tmp_user = {i:i for i in xrange(5)}
        map_tmp_item = {i:i for i in xrange(3)}
        preference_matrix = numpy.array([[ 0.69142586,  0.71043127,  0.69895359],
                                           [ 1.01069631,  0.79673869,  1.02444094],
                                           [ 0.48269756,  0.27936538,  0.4904087 ],
                                           [ 0.40879814,  0.26181097,  0.41504283],
                                           [ 0.58745062,  0.76717268,  0.59199168]])
        rd = RecommenderData(preference_matrix=preference_matrix,
                        map_idx2user=map_tmp_user,
                        map_idx2item=map_tmp_item,
                        map_user2idx=map_tmp_user,
                        map_item2idx=map_tmp_item)

        model = SVD(rd)
        model.fit()

        retMtx = numpy.dot(model.user_matrix, model.item_matrix)
        self.assertTrue(numpy.allclose(preference_matrix, retMtx, 1.0e-5, 1.0e-5))
        model.get_score()

    def test_lda(self):

        map_tmp_user = {i:i for i in xrange(5)}
        map_tmp_item = {i:i for i in xrange(3)}
        preference_matrix = numpy.array([[ 0.69142586,  0.71043127,  0.69895359],
                                           [ 1.01069631,  0.79673869,  1.02444094],
                                           [ 0.48269756,  0.27936538,  0.4904087 ],
                                           [ 0.40879814,  0.26181097,  0.41504283],
                                           [ 0.58745062,  0.76717268,  0.59199168]])
        rd = RecommenderData(preference_matrix=preference_matrix,
                        map_idx2user=map_tmp_user,
                        map_idx2item=map_tmp_item,
                        map_user2idx=map_tmp_user,
                        map_item2idx=map_tmp_item)

        model = LDA(rd)
        model.fit()

        print model.get_score()

    """
    def test_bpr(self):

        map_tmp_user = {i:i for i in xrange(5)}
        map_tmp_item = {i:i for i in xrange(3)}
        preference_matrix = numpy.array([[ 0.69142586,  0.71043127,  0.69895359],
                                           [ 1.01069631,  0.79673869,  1.02444094],
                                           [ 0.48269756,  0.27936538,  0.4904087 ],
                                           [ 0.40879814,  0.26181097,  0.41504283],
                                           [ 0.58745062,  0.76717268,  0.59199168]])
        rd = RecommenderData(preference_matrix=preference_matrix,
                        map_idx2user=map_tmp_user,
                        map_idx2item=map_tmp_item,
                        map_user2idx=map_tmp_user,
                        map_item2idx=map_tmp_item)

        model = BPR(rd)
        model.fit()

        print model.get_score()
