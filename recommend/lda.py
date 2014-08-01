#!/usr/bin/env python
# -*- coding:utf-8

from recommender import Recommender
import numpy
import pandas
import scipy.sparse as spMat

from gensim.matutils import Sparse2Corpus, corpus2dense
from gensim.models.ldamodel import LdaModel
class LDA(Recommender):

    def fit(self, k=100, max_iter=15, method='lsnmf'):
        if self.recommender_data.preference_matrix.shape[1] < k:
            k = self.recommender_data.preference_matrix.shape[1]

        if not spMat.isspmatrix_lil(self.recommender_data.preference_matrix):
            self.recommender_data.preference_matrix = spMat.lil_matrix(self.recommender_data.preference_matrix)
        self.perp = None

        model = LDA_CVB0(self.recommender_data.preference_matrix, K=k)
        model.lda_learning(max_iter)

        self.user_matrix = model.documentdist()
        self.item_matrix = model.worddist()


# Latent Dirichlet Allocation + Collapsed Variational Bayesian
# This code is available under the MIT License.
# (c)2010-2011 Nakatani Shuyo / Cybozu Labs Inc.
class LDA_CVB0:
    def __init__(self, docs, K, alpha=0.5, beta=0.5):

        self.K = K
        self.alpha = alpha
        self.beta = beta
        V = docs.shape[1]
        self.V = docs.shape[1]

        self.docs = []
        self.gamma_jik = []
        self.n_wk = numpy.zeros((V, K)) + beta
        self.n_jk = numpy.zeros((docs.shape[0], K)) + alpha
        self.n_k = numpy.zeros(K) + V * beta

        for i in xrange(docs.shape[0]):
            term_freq = dict()
            term_gamma = dict()
            doc = docs[i]
            for j in xrange(doc.nnz):
                gamma_k = numpy.random.mtrand.dirichlet([alpha] * K)
                num = doc.data[0][j]
                w = doc.rows[0][j]

                term_freq[w] = num
                term_gamma[w] = gamma_k * num

                self.n_wk[w] = gamma_k * num
                self.n_jk[i] = gamma_k * num
                self.n_k = gamma_k * num
            term_freq = term_freq.items()
            self.docs.append(term_freq)
            self.gamma_jik.append([term_gamma[w] / freq for w, freq in term_freq])

    def inference(self):
        """learning once iteration"""
        new_n_wk = numpy.zeros((self.V, self.K)) + self.beta
        new_n_jk = numpy.zeros((len(self.docs), self.K)) + self.alpha
        n_k = self.n_k
        for j, doc in enumerate(self.docs):
            gamma_ik = self.gamma_jik[j]
            n_jk = self.n_jk[j]
            new_n_jk_j = new_n_jk[j]
            for i, gamma_k in enumerate(gamma_ik):
                w, freq = doc[i]
                new_gamma_k = (self.n_wk[w] - gamma_k) * (n_jk - gamma_k) / (n_k - gamma_k)
                new_gamma_k /= new_gamma_k.sum()

                gamma_ik[i] = new_gamma_k
                gamma_freq = new_gamma_k * freq
                new_n_wk[w] += gamma_freq
                new_n_jk_j += gamma_freq

        self.n_wk = new_n_wk
        self.n_jk = new_n_jk
        self.n_k  = new_n_wk.sum(axis=0)

    def worddist(self):
        """get topic-word distribution"""
        return numpy.transpose(self.n_wk / self.n_k)

    def documentdist(self):
        """get document-word distribution"""
        return numpy.array([row/row.sum() for row in self.n_jk])

    def perplexity(self, docs=None):
        if docs == None: docs = self.docs
        phi = self.worddist()
        log_per = 0
        N = 0
        for j, doc in enumerate(docs):
            theta = self.n_jk[j]
            theta = theta / theta.sum()
            for w, freq in doc:
                log_per -= numpy.log(numpy.inner(phi[:,w], theta)) * freq
                N += freq
        return numpy.exp(log_per / N)

    def lda_learning(self, iteration):
        for _ in range(iteration):
            self.inference()

        self.perp = self.perplexity()


