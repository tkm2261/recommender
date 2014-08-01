#!/usr/bin/env python
# -*- coding:utf-8

from recommender import Recommender
import nimfa
import numpy
import pandas

class NMF(Recommender):

    def fit(self, k=100, max_iter=15, method='lsnmf'):
        if self.recommender_data.preference_matrix.shape[1] < k:
            k = self.recommender_data.preference_matrix.shape[1]


        model = nimfa.mf(self.recommender_data.preference_matrix,
                            seed="random_vcol",
                            rank=k,
                            method=method,
                            max_iter=max_iter)

        fit = nimfa.mf_run(model)

        self.user_matrix = fit.basis().todense()
        self.item_matrix = fit.coef().todense()




