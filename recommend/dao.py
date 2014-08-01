#!/usr/bin/env python
# -*- coding:utf-8
import scipy.sparse as spMat
import pandas
import numpy
from io import RecommenderData

COLUMN_NAMES=["user", "item", "score"]

def read_csv(path, header=0, delimiter=","):

    try:
        data = pandas.read_csv(path,
                               sep=delimiter,
                               header=header,
                               names=COLUMN_NAMES)
    except:
        raise Exception("failed to open file {}".format(path))

    user_ids = numpy.sort(data["user"].unique())
    item_ids = numpy.sort(data["item"].unique())

    map_idx2user = dict([(i, user_ids[i]) for i in xrange(len(user_ids))])
    map_idx2item = dict([(i, item_ids[i]) for i in xrange(len(item_ids))])
    map_user2idx = dict([(user_ids[i], i) for i in xrange(len(user_ids))])
    map_item2idx = dict([(item_ids[i], i) for i in xrange(len(item_ids))])

    data["user"] = data["user"].apply(lambda x:map_user2idx[x])
    data["item"] = data["item"].apply(lambda x:map_item2idx[x])


    data = spMat.coo_matrix(
                     (data["score"], (data["user"], data["item"])),
                      shape=(len(user_ids), len(item_ids)),
                     dtype=numpy.double
                      )

    return RecommenderData(data,
                             map_idx2user,
                             map_idx2item,
                             map_user2idx,
                             map_item2idx)
