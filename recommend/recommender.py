#!/usr/bin/env python
# -*- coding:utf-8
import scipy.sparse as spMat

class Recommender(object):

    def __init__(self,
                 preference_matrix,
                 map_idx2user=None,
                 map_idx2item=None,
                 ):

        if not spMat.issparse(preference_matrix):
            try:
                self.preference_matrix = spMat.csr_matrix(preference_matrix)
            except:
                raise TypeError("preference matrix cannot convert sparse format")
        else:
            self.preference_matrix = preference_matrix

        self.map_idx2user = map_idx2user
        self.map_idx2item = map_idx2item

        self.user_matrix = None
        self.item_matrix = None
        self.user_item_link_matrix = None


    def fit(self):
        pass

    def predict(self):
        pass

    def get_all_score(self):
        pass
