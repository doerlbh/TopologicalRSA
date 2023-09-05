#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# functions related to statistical inference
# author: baihan lin (doerlbh@gmail.com)


import numpy as np
from scipy.spatial.distance import squareform
from scipy import stats
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score
from scipy.spatial import distance_matrix
from sklearn.impute import SimpleImputer

clf = NearestCentroid()
imp = SimpleImputer(missing_values=np.nan, strategy="mean")


def bootstrap(
    rdms,
    subjs,
    rois,
    boot_type="both",
    n_boot_subjs=None,
    n_boot_stims=None,
    subj_boots=None,
    stim_boots=None,
    **kwargs
):
    n_sess = len(subjs)
    n_subjs = len(np.unique(subjs))
    n_stims = squareform(rdms[0]).shape[0]
    n_boot_subjs = n_boot_subjs or len(subjs)
    n_boot_stims = n_boot_stims or n_stims
    if boot_type in ["both", "subj"]:
        selected_subjs = (
            subj_boots
            if subj_boots is not None
            else np.random.choice(n_sess, n_boot_subjs, replace=True)
        )
    else:
        selected_subjs = np.arange(n_subjs)
    if boot_type in ["both", "stim"]:
        selected_stims = (
            stim_boots
            if stim_boots is not None
            else np.random.choice(n_stims, n_boot_stims, replace=True)
        )
    else:
        selected_stims = np.arange(n_stims)
    selected_rdms = rdms[selected_subjs]
    bootstrapped_rdms = np.array(
        [bootstrap_rdm_by_stims(r, selected_stims) for r in selected_rdms]
    )
    bootstrapped_subjs = subjs[selected_subjs]
    bootstrapped_rois = rois[selected_subjs]
    return bootstrapped_rdms, bootstrapped_subjs, bootstrapped_rois


def bootstrap_rdm_by_stims(r, selected_stims):
    bootstrapped_rdm = -np.ones((len(selected_stims), len(selected_stims)))
    rdm = squareform(r)
    for i, s1 in enumerate(selected_stims):
        for j, s2 in enumerate(selected_stims):
            if i == j:
                bootstrapped_rdm[i, j] = bootstrapped_rdm[j, i] = 0
            else:
                if s1 != s2:
                    bootstrapped_rdm[i, j] = bootstrapped_rdm[j, i] = rdm[s1, s2]
    bootstrapped = squareform(bootstrapped_rdm)
    bootstrapped[bootstrapped == -1] = np.nan
    return list(bootstrapped)


def inner_boots(params):
    i, boots_seq, rdms, subjs, rois, roi_names, handle_duplicate, n_boots = params
    if boots_seq is not None and len(boots_seq) == n_boots:
        subj_boots = boots_seq[i]["subj_boots"]
        stim_boots = boots_seq[i]["stim_boots"]
    else:
        subj_boots = None
        stim_boots = None
    b_rdms, b_subjs, b_rois = bootstrap(
        rdms, subjs, rois, subj_boots=subj_boots, stim_boots=stim_boots
    )
    if handle_duplicate == "mean":
        b_rdms = imp.fit_transform(b_rdms)  # for mean imputation
    elif handle_duplicate == "drop":
        b_rdms = b_rdms[:, ~np.isnan(b_rdms).any(axis=0)]  # for drop nan
    acc, _, _ = loo_eval(roi_names, b_rdms, b_subjs, b_rois)
    return acc


def get_boots_estimates(
    roi_names,
    rdms,
    subjs,
    rois,
    n_boots=10,
    handle_duplicate="drop",
    boots_seq=None,
    **kwargs
):
    def params_wrapper(n):
        return [
            (i, boots_seq, rdms, subjs, rois, roi_names, handle_duplicate, n_boots)
            for i in range(n)
        ]

    boots_accs = process_map(inner_boots, params_wrapper(n_boots))
    return boots_accs


def loo_eval(
    roi_names, rdms, subjs, rois, eval_method="loo", additional_stats=False, **kwargs
):
    rdms, subjs, rois = (
        np.array(rdms),
        np.array(subjs).squeeze(),
        np.array(rois).squeeze(),
    )
    unique_subj = np.unique(np.array(subjs))
    unique_roi = np.unique(np.array(rois))
    acc, d_vals, d_pairs = [], [], []
    d_subjs, d_rois = [], []
    d_ratios_by_subjs, d_rois_by_subjs = [], []
    if eval_method == "loo":
        label_set, pred_set = [], []
        for s in unique_subj:
            test_set = rdms[subjs == s]
            test_label = rois[subjs == s]
            train_set = rdms[subjs != s]
            train_label = rois[subjs != s]
            clf.fit(train_set, train_label)
            label_set += list(test_label)
            pred_set += list(clf.predict(test_set))
            if additional_stats:
                for r in unique_roi:
                    train_r, train_nr = (
                        train_set[train_label == r],
                        train_set[train_label != r],
                    )
                    test_r, test_nr = (
                        test_set[test_label == r],
                        test_set[test_label != r],
                    )
                    ds = [np.mean(distance_matrix(test_r, train_r).flatten())]
                    d_vals += list(ds)
                    within = ds[0]
                    d_pairs += ["within"] * len(ds)
                    d_subjs += [s] * len(ds)
                    d_rois += [roi_names[r - 1]] * len(ds)
                    ds = [np.mean(distance_matrix(test_r, train_nr).flatten())]
                    d_vals += list(ds)
                    across = ds[0]
                    d_pairs += ["across"] * len(ds)
                    d_subjs += [s] * len(ds)
                    d_rois += [roi_names[r - 1]] * len(ds)
                    d_ratios_by_subjs.append(within / across)
                    d_rois_by_subjs.append(roi_names[r - 1])
                stats_d, stats_r = {
                    "distance": d_vals,
                    "distance_type": d_pairs,
                    "subject": d_subjs,
                    "roi": d_rois,
                    "length": len(d_vals),
                }, {"ratio": d_ratios_by_subjs, "roi": d_rois_by_subjs}
            else:
                stats_d, stats_r = None, None
        acc = accuracy_score(label_set, pred_set)
        return acc, stats_d, stats_r


def get_boots_seq(
    n_boots=10, n_subjs=10, n_stims=10, n_sess=10, n_boot_subjs=None, n_boot_stims=None
):
    boots_seq = []
    n_boot_subjs = n_boot_subjs or n_sess
    n_boot_stims = n_boot_stims or n_stims
    for n in range(n_boots):
        boot_seq = {
            "subj_boots": np.random.choice(n_sess, n_boot_subjs, replace=True),
            "stim_boots": np.random.choice(n_stims, n_boot_stims, replace=True),
        }
        boots_seq.append(boot_seq)
    return boots_seq


def get_paired_p_values(md, s, p_df):
    t = md / s
    p = stats.t.sf(np.abs(t), p_df) * 2
    return p
