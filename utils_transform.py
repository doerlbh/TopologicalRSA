#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# functions related to GT transform
# author: baihan lin (doerlbh@gmail.com)


import numpy as np
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
import networkx as nx
from scipy.cluster.hierarchy import linkage
from operator import itemgetter


def minmax_transform(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def rank_transform(data):
    return rankdata(data)


def gt_transform(data, gt_type="lu", **kwargs):
    data = minmax_transform(rank_transform(data))
    gtx = None
    if gt_type == "lu":
        l = kwargs.get("l", 0.2)
        u = kwargs.get("u", 0.8)
        gt_min = np.quantile(data, l)
        gt_max = np.quantile(data, u)
        gtx = data.copy()
        gtx[data < gt_min] = 0
        gtx[(data >= gt_min) & (data <= gt_max)] = (
            data[(data >= gt_min) & (data <= gt_max)] - gt_min
        ) / (gt_max - gt_min)
        gtx[data > gt_max] = 1
    elif gt_type == "sigmoid":
        alpha = kwargs.get("alpha", 10)
        beta = kwargs.get("beta", 0.5)
        gtx = 1 / (1 + np.exp(-(data - beta) * alpha))
    else:
        print("please specify the right gt_type")
    return gtx


def geodesic_transform(data, gd_type="p_shortest", rank_transform_after=True, **kwargs):
    if gd_type == "twostep":
        data = minmax_transform(data)
    else:
        data = minmax_transform(rank_transform(data))
    G = nx.from_numpy_array(squareform(data))
    long_edges = []
    if gd_type == "twostep":
        gtx = data.copy()
        long_edges = list(
            filter(lambda e: e[2] == 1, (e for e in G.edges.data("weight")))
        )
    elif gd_type == "p_shortest":
        p = kwargs.get("p", 0.05)
        threshold = np.quantile(data, p)
        gtx = data.copy()
        long_edges = list(
            filter(lambda e: e[2] > threshold, (e for e in G.edges.data("weight")))
        )
    elif gd_type == "p_nearest":
        p = kwargs.get("p", 0.05)
        k = int(p * len(G.nodes()))
        long_edges = list(
            filter(
                lambda e: e[1] not in knn(G, e[0], k),
                (e for e in G.edges.data("weight")),
            )
        )
    elif gd_type == "k_nearest":
        k = kwargs.get("k", 1)
        long_edges = list(
            filter(
                lambda e: e[1] not in knn(G, e[0], k),
                (e for e in G.edges.data("weight")),
            )
        )
    else:
        print("please specify the right gd_type")
    le_ids = list(e[:2] for e in long_edges)
    G.remove_edges_from(le_ids)
    gtx = squareform(np.array(nx.floyd_warshall_numpy(G)))
    if rank_transform_after:
        gtx = rank_transform(gtx)
    return gtx


def knn(graph, node, n):
    return list(
        map(
            itemgetter(1),
            sorted([(e[2]["weight"], e[1]) for e in graph.edges(node, data=True)])[:n],
        )
    )


def matrix_ranktransform(data):
    return minmax_transform(rank_transform(data)).reshape(np.shape(data))


def finitemax(a):
    a = np.array(a)
    return a[np.isfinite(a)].max()


def sort_dist(m):
    perm = np.random.permutation(len(m))
    m = m[perm][:, perm]
    y = m[np.triu_indices(len(m), k=1)]
    Z = linkage(y, method="single", optimal_ordering=True)
    perm = np.ravel(Z[:, :2]).astype(np.int32)
    perm = perm[perm < len(m)]
    m = m[perm][:, perm]
    return m
