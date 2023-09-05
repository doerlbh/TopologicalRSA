#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# functions to reproduce the results in the paper
# author: baihan lin (doerlbh@gmail.com)

from utils_transform import *
from utils_stats import *

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
from sklearn.manifold import MDS
from scipy.spatial import ConvexHull
from sklearn import datasets as skds
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import tadasets
from scipy.spatial import procrustes
from scipy.io import loadmat
from sklearn.neighbors import NearestCentroid
from statannotations.Annotator import Annotator
from matplotlib.patches import Rectangle
import itertools
from scipy.interpolate import interp1d
from matplotlib.colors import ListedColormap

# from mayavi import mlab

clf = NearestCentroid()

mds = MDS(n_components=2)
nmds = MDS(n_components=2, metric=False)
pmds = MDS(n_components=2, dissimilarity="precomputed")

topo_patterns = ["torus", "swiss", "s_curve", "line", "sphere", "gaussian"]
cate_patterns = ["classification", "blobs", "gaussian_quantiles"]
ring_patterns = [
    "flat_ring",
    "twist_ring",
    "double_twist_ring",
    "two_rings",
    "mixed_rings",
    "locked_rings",
]
eigt_patterns = [
    "flat_8",
    "twist_8",
    "broken_8",
    "double_twist_8",
    "anti_double_twist_8",
    "triple_twist_8",
    "anti_triple_twist_8",
]
patterns = topo_patterns + cate_patterns + ring_patterns + eigt_patterns

RGTM_NAMES = [
    "topology sensitive",
    "geometry sensitive",
    "local extractor",
    "global extractor",
    "intermediate",
]

RGTM_NAMES_RDM = [
    "RDM",
    "RGTM (topology sensitive)",
    "RGTM (geometry sensitive)",
    "RGTM (local extractor)",
    "RGTM (global extractor)",
    "RGTM (intermediate)",
]


def get_dataset(dataset_name="trans62", env="mac"):
    if env == "mac":
        folder = "/Users/doerlbh"
    elif env == "linux":
        folder = "/home/doerlbh"
    if dataset_name == "trans62":
        mat = loadmat(f"{folder}/Dropbox/Git/RGTA/trans62_avg_data.mat")
        mdata = mat["D"]
        mtype = mdata.dtype
        ndata = {n: mdata[n][0, 0] for n in mtype.names}
        roi_names = ["V1 ", "V2 ", "V3 ", "LOC", "OFA", "FFA", "PPA", "aIT"]
        rdms, subjs, rois = ndata["rdm"], ndata["subj"], ndata["roi"]

    elif "cnn62_0p" in dataset_name:
        allrdms, allnoises, allsubjects, allrois = [], [], [], []
        for n in np.arange(10):
            for s in np.arange(10):
                fname = f"{folder}/Dropbox/Git/RGTA/newdnn_eval/models_varying_noise/ReLU/blt-rcnn_all_cnn_c_350_epochs_noiselevel_0p{n}_training_seed_{str(int(s+1))}/activations/rdms.npy"
                combined_rdms = np.load(fname)
                for l in np.arange(10):
                    allrdms.append(list(combined_rdms[l].flatten()))
                    allnoises.append(n / 10)
                    allsubjects.append(s)
                    allrois.append(l + 1)
        roi_names = [
            "Layer 1 ",
            "Layer 2 ",
            "Layer 3 ",
            "Layer 4 ",
            "Layer 5 ",
            "Layer 6 ",
            "Layer 7 ",
            "Layer 8 ",
            "Layer 9 ",
            "Layer 10",
        ]
        allrdms, allnoises, allsubjects, allrois = (
            np.array(allrdms),
            np.array(allnoises),
            np.array(allsubjects),
            np.array(allrois),
        )

        noise_degree = int(dataset_name[-1])
        selected = allnoises == noise_degree / 10
        rdms, subjs, rois = allrdms[selected], allsubjects[selected], allrois[selected]

    elif "cnn62_g_0p" in dataset_name:
        allrdms, allnoises, allsubjects, allrois = [], [], [], []
        for n in np.arange(10):
            for s in np.arange(10):
                fname = f"{folder}/Dropbox/Git/RGTA/newdnn_eval/models_varying_noise/ReLU/blt-rcnn_all_cnn_c_350_epochs_noiselevel_0p{n}_training_seed_{str(int(s+1))}/activations/rdms.npy"
                combined_rdms = np.load(fname)
                for l in np.arange(10):
                    allrdms.append(list(combined_rdms[l].flatten()))
                    allnoises.append(n / 10)
                    allsubjects.append(s)
                    allrois.append(l + 1)
        roi_names = [
            "Layer 1 ",
            "Layer 2 ",
            "Layer 3 ",
            "Layer 4 ",
            "Layer 5 ",
            "Layer 6 ",
            "Layer 7 ",
            "Layer 8 ",
            "Layer 9 ",
            "Layer 10",
        ]
        allrdms, allnoises, allsubjects, allrois = (
            np.array(allrdms),
            np.array(allnoises),
            np.array(allsubjects),
            np.array(allrois),
        )

        noise_degree = int(dataset_name[-1])
        selected = allnoises == 0
        rdms, subjs, rois = allrdms[selected], allsubjects[selected], allrois[selected]
        rdms += np.random.normal(
            loc=0, scale=np.std(rdms) * noise_degree / 10, size=rdms.shape
        )
        rdms[rdms < 0] = 0

    elif "cnn_cifar_0p" in dataset_name:
        mat = loadmat(
            f"{folder}/Dropbox/Git/RGTA/20181115_RDM_consistency/cifar10_dnn.mat"
        )
        mdata = mat["D"]
        mtype = mdata.dtype
        ndata = {n: mdata[n][0, 0] for n in mtype.names}
        roi_names = [
            "Layer 1 ",
            "Layer 2 ",
            "Layer 3 ",
            "Layer 4 ",
            "Layer 5 ",
            "Layer 6 ",
            "Layer 7 ",
            "Layer 8 ",
            "Layer 9 ",
            "Layer 10",
        ]
        noise_degree = int(dataset_name[-1])
        rdms = ndata["rdm"][np.nonzero(ndata["expid"] == noise_degree + 9)[0], :]
        rois = ndata["roi"][np.nonzero(ndata["expid"] == noise_degree + 9)[0], :]
        subjs = ndata["subj"][np.nonzero(ndata["expid"] == noise_degree + 9)[0], :]

    rdms, subjs, rois = np.array(rdms), np.array(subjs), np.array(rois)

    return roi_names, rdms, subjs, rois


def get_dataset_configs(rdms, subjs, rois):
    n_sess = len(subjs)
    n_subjs = len(np.unique(subjs))
    n_stims = squareform(rdms[0]).shape[0]
    n_rois = len(np.unique(rois))
    return n_sess, n_subjs, n_stims, n_rois


def get_shape(shape="torus", sample=1000, **kwargs):
    data = None
    supp = np.arange(sample)

    noise = kwargs.get("noise", 0.0)
    if shape == "torus":
        data = tadasets.torus(n=sample, c=10, a=4, noise=noise)
    elif shape == "swiss":
        data = tadasets.swiss_roll(n=sample, r=10, noise=noise)
    elif shape == "s_curve":
        data, supp = skds.make_s_curve(n_samples=sample, noise=noise)
    elif shape == "sphere":
        data = tadasets.sphere(n=sample, r=10, noise=noise)
    elif shape == "line":
        data = np.concatenate(
            [
                (np.arange(sample) / sample).reshape((-1, 1)),
                (np.arange(sample) / sample).reshape((-1, 1)),
                (np.arange(sample) / sample).reshape((-1, 1)),
            ],
            axis=1,
        )
    elif shape == "gaussian":
        mean = [0, 0, 0]
        cov = [[noise, 0, 0], [0, noise, 0], [0, 0, noise]]
        data = np.random.multivariate_normal(mean, cov, sample)
    elif shape == "classification":
        n_redundant = kwargs.get("n_redundant", 0)
        n_informative = kwargs.get("n_informative", 3)
        n_clusters_per_class = kwargs.get("n_clusters_per_class", 1)
        n_features = kwargs.get("n_features", 3)
        n_classes = kwargs.get("n_classes", 2)
        data, supp = skds.make_classification(
            n_samples=sample,
            n_redundant=n_redundant,
            n_informative=n_informative,
            n_features=n_features,
            n_clusters_per_class=n_clusters_per_class,
            n_classes=n_classes,
            random_state=0,
        )
    elif shape == "blobs":
        n_features = kwargs.get("n_features", 3)
        n_classes = kwargs.get("n_classes", 2)
        data, supp = skds.make_blobs(
            n_samples=sample, centers=n_classes, n_features=n_features, random_state=0
        )
    elif shape == "gaussian_quantiles":
        n_features = kwargs.get("n_features", 3)
        n_classes = kwargs.get("n_classes", 2)
        data, supp = skds.make_gaussian_quantiles(
            cov=noise,
            n_samples=sample,
            n_features=n_features,
            n_classes=n_classes,
            random_state=0,
        )
    elif shape == "two_rings":
        n_features = kwargs.get("n_features", 3)
        n_classes = kwargs.get("n_classes", 2)
        factor = kwargs.get("factor", 0.5)
        x, supp = skds.make_circles(
            n_samples=sample, noise=noise, factor=factor, random_state=0
        )
        data = np.concatenate([x, np.zeros((sample, 1))], axis=1)
    elif shape == "mixed_rings":
        n_ring1 = int(sample / 2)
        n_ring2 = sample - n_ring1
        ring1 = np.concatenate(
            [
                np.sin(np.arange(n_ring1) * 2 * np.pi / n_ring1).reshape((-1, 1)),
                np.cos(np.arange(n_ring1) * 2 * np.pi / n_ring1).reshape((-1, 1)),
                0 * np.ones((n_ring1, 1)),
            ],
            axis=1,
        )
        ring2 = np.concatenate(
            [
                0 * np.ones((n_ring2, 1)),
                np.sin(np.arange(n_ring2) * 2 * np.pi / n_ring2).reshape((-1, 1)),
                np.cos(np.arange(n_ring2) * 2 * np.pi / n_ring2).reshape((-1, 1)),
            ],
            axis=1,
        )
        data = np.concatenate([ring1, ring2])
    elif shape == "locked_rings":
        n_ring1 = int(sample / 2)
        n_ring2 = sample - n_ring1
        ring1 = np.concatenate(
            [
                np.sin(np.arange(n_ring1) * 2 * np.pi / n_ring1).reshape((-1, 1)),
                1 + np.cos(np.arange(n_ring1) * 2 * np.pi / n_ring1).reshape((-1, 1)),
                0 * np.ones((n_ring1, 1)),
            ],
            axis=1,
        )
        ring2 = np.concatenate(
            [
                0 * np.ones((n_ring2, 1)),
                np.sin(np.arange(n_ring2) * 2 * np.pi / n_ring2).reshape((-1, 1)),
                np.cos(np.arange(n_ring2) * 2 * np.pi / n_ring2).reshape((-1, 1)),
            ],
            axis=1,
        )
        data = np.concatenate([ring1, ring2])
    elif shape == "flat_ring":
        data = np.concatenate(
            [
                np.sin(np.arange(sample) * 2 * np.pi / sample).reshape((-1, 1)),
                np.cos(np.arange(sample) * 2 * np.pi / sample).reshape((-1, 1)),
                np.zeros((sample, 1)).reshape((-1, 1)),
            ],
            axis=1,
        )
    elif shape == "twist_ring":
        data = np.concatenate(
            [
                np.sin(np.arange(sample) * 2 * np.pi / sample).reshape((-1, 1)),
                np.cos(np.arange(sample) * 2 * np.pi / sample).reshape((-1, 1)),
                np.sin(np.arange(sample) * 4 * np.pi / sample).reshape((-1, 1)),
            ],
            axis=1,
        )
    elif shape == "double_twist_ring":
        data = np.concatenate(
            [
                np.sin(np.arange(sample) * 2 * np.pi / sample).reshape((-1, 1)),
                np.cos(np.arange(sample) * 2 * np.pi / sample).reshape((-1, 1)),
                np.cos(np.arange(sample) * 8 * np.pi / sample).reshape((-1, 1)),
            ],
            axis=1,
        )
    elif shape == "flat_8":
        infty = tadasets.infty_sign(n=sample, noise=noise)
        data = np.concatenate([infty, np.zeros((sample, 1)).reshape((-1, 1))], axis=1)
    elif shape == "twist_8":
        infty = tadasets.infty_sign(n=sample, noise=noise)
        data = np.concatenate(
            [
                infty / 2,
                2 * np.cos(np.arange(sample) * np.pi * 4 / sample).reshape((-1, 1)) / 2,
            ],
            axis=1,
        )
    elif shape == "untouched_flat_8":
        infty = tadasets.infty_sign(n=sample, noise=noise)
        z = (
            np.cos(np.arange(sample) * np.pi * 2 / sample - np.pi / 2).reshape((-1, 1))
            / 3
        )
        data = np.concatenate([infty, z], axis=1)
    elif shape == "untouched_twist_8":
        infty = tadasets.infty_sign(n=sample, noise=noise)
        z = np.multiply(
            np.cos(np.arange(sample) * np.pi * 4 / sample) / 2,
            (np.sin(np.arange(sample) * 2 * np.pi / sample) / 3 + 1),
        ).reshape((-1, 1))
        data = np.concatenate([infty / 2, 2 * z], axis=1)
    elif shape == "broken_8":
        infty = tadasets.infty_sign(n=sample, noise=noise)
        data = np.concatenate(
            [infty, np.cos(np.arange(sample) * np.pi / sample).reshape((-1, 1))], axis=1
        )
    elif shape == "double_twist_8":
        infty = tadasets.infty_sign(n=sample, noise=noise)
        data = np.concatenate(
            [infty, np.cos(np.arange(sample) * 8 * np.pi / sample).reshape((-1, 1))],
            axis=1,
        )
    elif shape == "anti_double_twist_8":
        infty = tadasets.infty_sign(n=sample, noise=noise)
        data = np.concatenate(
            [infty, np.sin(np.arange(sample) * 8 * np.pi / sample).reshape((-1, 1))],
            axis=1,
        )
    elif shape == "triple_twist_8":
        infty = tadasets.infty_sign(n=sample, noise=noise)
        data = np.concatenate(
            [infty, np.cos(np.arange(sample) * 16 * np.pi / sample).reshape((-1, 1))],
            axis=1,
        )
    elif shape == "anti_triple_twist_8":
        infty = tadasets.infty_sign(n=sample, noise=noise)
        data = np.concatenate(
            [infty, np.sin(np.arange(sample) * 16 * np.pi / sample).reshape((-1, 1))],
            axis=1,
        )
    else:
        print("please specify the right shape.")
    return data, supp


def get_selected_patterns(patterns, sample=40):
    patterns_dict = {}
    for p in patterns:
        data, supp = get_shape(shape=p, sample=sample)
        patterns_dict[p] = {"x": data, "y": supp}
    return patterns_dict


def get_all_patterns(patterns, sample=100):
    patterns_dict = {}
    for p in patterns:
        data, supp = get_shape(shape=p, sample=sample)
        patterns_dict[p] = {"x": data, "y": supp}
    return patterns_dict


def get_gt_df(x, y=None, search_type="grid", **kwargs):
    data = gt_parameter_search(x, y, search_type, **kwargs)
    dr_x = np.nan_to_num(data["rdm"])
    sorted_dr_x = np.nan_to_num(data["sorted_rdm"])
    sorted_rdm_mds = mds.fit_transform(sorted_dr_x)
    idv_rdm_mds = np.array([pmds.fit_transform(squareform(rdm)) for rdm in dr_x])
    rdm_mds = mds.fit_transform(dr_x)
    N = len(data["l"])
    df = pd.DataFrame(
        {
            "dataset": data["dataset"],
            "mds_0": list(idv_rdm_mds[:N, :, 0]),
            "mds_1": list(idv_rdm_mds[:N, :, 1]),
            "full_mds_0": rdm_mds[:N, 0],
            "full_mds_1": rdm_mds[:N, 1],
            "sorted_mds_0": sorted_rdm_mds[:N, 0],
            "sorted_mds_1": sorted_rdm_mds[:N, 1],
            "rdm": data["rdm"],
            "sorted_rdm": data["sorted_rdm"],
            "l": data["l"],
            "u": data["u"],
            "p": data["p"],
            "k": data["k"],
            "rdm_type": data["rdm_type"],
        }
    )
    return df, data


def get_gd_df(datasets_dict, search_type="grid", **kwargs):
    data = {
        "rdm": [],
        "p": [],
        "k": [],
        "l": [],
        "u": [],
        "sorted_rdm": [],
        "dataset": [],
        "rdm_type": [],
    }
    for k, v in datasets_dict.items():
        print("====", k, "====")
        data = gd_parameter_search(
            x=v["x"],
            y=v["y"],
            search_type=search_type,
            dataset_name=k,
            dictionary=data,
            **kwargs,
        )
    dr_x = np.nan_to_num(data["rdm"], posinf=finitemax(data["rdm"]))
    sorted_dr_x = np.nan_to_num(
        data["sorted_rdm"], posinf=finitemax(data["sorted_rdm"])
    )
    rdm_mds = nmds.fit_transform(dr_x)
    sorted_rdm_mds = nmds.fit_transform(sorted_dr_x)
    idv_rdm_mds = np.array([pmds.fit_transform(squareform(rdm)) for rdm in dr_x])
    N = len(data["rdm"])
    df = pd.DataFrame(
        {
            "dataset": data["dataset"],
            "mds_0": list(idv_rdm_mds[:N, :, 0]),
            "mds_1": list(idv_rdm_mds[:N, :, 1]),
            "full_mds_0": rdm_mds[:N, 0],
            "full_mds_1": rdm_mds[:N, 1],
            "sorted_mds_0": sorted_rdm_mds[:N, 0],
            "sorted_mds_1": sorted_rdm_mds[:N, 1],
            "rdm": data["rdm"],
            "l": data["l"],
            "u": data["u"],
            "p": data["p"],
            "k": data["k"],
            "sorted_rdm": data["sorted_rdm"],
            "rdm_type": data["rdm_type"],
        }
    )
    return df, data


def get_rgtm_type(l, u):
    if np.mean([l, u]) <= 0.2 + 1e-3:
        rgtm_here = "RGTM (local extractor)"
    elif np.mean([l, u]) >= 0.8 - 1e-3:
        rgtm_here = "RGTM (global extractor)"
    elif np.abs(u - l) >= 0.8 - 1e-3:
        rgtm_here = "RGTM (geometry sensitive)"
    elif (l >= 0.4 - 1e-3) & (u <= 0.6 + 1e-3):
        rgtm_here = "RGTM (topology sensitive)"
    else:
        rgtm_here = "RGTM (intermediate)"
    return rgtm_here


def get_dataset_df(
    datasets_dict, gt_search_type="grid", gd_search_type="grid", gt_type="gt", **kwargs
):
    data = {
        "rdm": [],
        "p": [],
        "k": [],
        "l": [],
        "u": [],
        "sorted_rdm": [],
        "dataset": [],
        "rdm_type": [],
    }
    for k, v in datasets_dict.items():
        print("====", k, "====")
        if gt_type in ["gt", "both", "all"]:
            print("--- gt ---")
            data = gt_parameter_search(
                x=v["x"],
                search_type=gt_search_type,
                dataset_name=k,
                dictionary=data,
                **kwargs,
            )
            print("- gt done")
        if gt_type in ["gd", "both", "all"]:
            print("--- gd ---")
            data = gd_parameter_search(
                x=v["x"],
                search_type=gd_search_type,
                dataset_name=k,
                dictionary=data,
                **kwargs,
            )
            print("- gd done")
        if gt_type in ["twostep", "all"]:
            print("--- twostep ---")
            l = kwargs.get("l", 0.0)
            u = kwargs.get("u", 1.0)
            data = gd_after_gt(x=v["x"], l=l, u=u, dataset_name=k, dictionary=data)
            print(f"- twostep done for l={l}, u={u}")
    dr_x = data["rdm"]
    sorted_dr_x = data["sorted_rdm"]
    dr_x = np.nan_to_num(dr_x, posinf=finitemax(data["rdm"]))
    sorted_dr_x = np.nan_to_num(sorted_dr_x, posinf=finitemax(data["sorted_rdm"]))
    rdm_mds = mds.fit_transform(dr_x)
    sorted_rdm_mds = mds.fit_transform(sorted_dr_x)
    idv_rdm_mds = np.array(
        [
            pmds.fit_transform(squareform(np.nan_to_num(rdm, posinf=finitemax(rdm))))
            for rdm in dr_x
        ]
    )
    N = len(data["l"])
    df = pd.DataFrame(
        {
            "dataset": data["dataset"],
            "mds_0": list(idv_rdm_mds[:N, :, 0]),
            "mds_1": list(idv_rdm_mds[:N, :, 1]),
            "full_mds_0": rdm_mds[:N, 0],
            "full_mds_1": rdm_mds[:N, 1],
            "sorted_mds_0": sorted_rdm_mds[:N, 0],
            "sorted_mds_1": sorted_rdm_mds[:N, 1],
            "rdm": data["rdm"],
            "sorted_rdm": data["sorted_rdm"],
            "l": data["l"],
            "u": data["u"],
            "p": data["p"],
            "k": data["k"],
            "rdm_type": data["rdm_type"],
        }
    )
    return df, data


def get_eval_df(
    datasets_dict, gt_search_type="grid", gd_search_type="single", **kwargs
):
    data = {
        "rdm": [],
        "p": [],
        "k": [],
        "l": [],
        "u": [],
        "sorted_rdm": [],
        "dataset": [],
        "rdm_type": [],
    }
    for k, v in datasets_dict.items():
        print("====", k, "====")
        if gt_search_type == "grid":
            include_RDM = True
        data = gt_parameter_search(
            x=v["x"],
            search_type=gt_search_type,
            dataset_name=k,
            dictionary=data,
            **kwargs,
        )
        data = gd_parameter_search(
            x=v["x"],
            search_type=gd_search_type,
            dataset_name=k,
            include_RDM=include_RDM,
            dictionary=data,
            **kwargs,
        )
    dr_x = np.nan_to_num(data["rdm"], posinf=finitemax(data["rdm"]))
    sorted_dr_x = np.nan_to_num(
        data["sorted_rdm"], posinf=finitemax((data["sorted_rdm"]))
    )
    idv_rdm_mds = np.array([pmds.fit_transform(squareform(rdm)) for rdm in dr_x])

    rdm_mds = mds.fit_transform(dr_x)
    sorted_rdm_mds = mds.fit_transform(sorted_dr_x)

    N = len(data["rdm"])
    df = pd.DataFrame(
        {
            "dataset": data["dataset"],
            "rdm": data["rdm"],
            "mds_0": list(idv_rdm_mds[:N, :, 0]),
            "mds_1": list(idv_rdm_mds[:N, :, 1]),
            "full_mds_0": rdm_mds[:N, 0],
            "full_mds_1": rdm_mds[:N, 1],
            "sorted_mds_0": sorted_rdm_mds[:N, 0],
            "sorted_mds_1": sorted_rdm_mds[:N, 1],
            "rdm": data["rdm"],
            "l": data["l"],
            "u": data["u"],
            "p": data["p"],
            "k": data["k"],
            "sorted_rdm": data["sorted_rdm"],
            "rdm_type": data["rdm_type"],
        }
    )
    return df, data


def gt_parameter_search(
    x, search_type="grid", dataset_name="data", dictionary=None, **kwargs
):
    if dictionary is None:
        data = {
            "l": [],
            "u": [],
            "p": [],
            "k": [],
            "rdm": [],
            "sorted_rdm": [],
            "dataset": [],
            "rdm_type": [],
        }
    else:
        data = dictionary
    N = x.shape[0]
    if search_type == "grid":
        resolution = kwargs.get("resolution", 0.05)
        resolution_min = kwargs.get("resolution_min", 0.0)
        resolution_max = kwargs.get("resolution_max", 1.0)
        for l in np.arange(resolution_min, resolution_max, resolution):
            for u in np.arange(l, resolution_max + 1e-4, resolution):
                l, u = np.min((l, 1)), np.min((u, 1))
                l, u = int(l * 100) / 100, int(u * 100) / 100
                if l == 0 and u == 1:
                    data["rdm_type"].append("RDM")
                else:
                    data["rdm_type"].append("RGTM")
                data["dataset"].append(dataset_name)
                data["l"].append(l)
                data["u"].append(u)
                data["p"].append(-1)
                data["k"].append(-1)
                data["rdm"].append(gt_transform(pdist(x), gt_type="lu", l=l, u=u))
                data["sorted_rdm"].append(
                    gt_transform(
                        squareform(sort_dist(squareform(pdist(x)))),
                        gt_type="lu",
                        l=l,
                        u=u,
                    )
                )
    elif search_type == "sample":
        sample = kwargs.get("sample", 100)
        for l in np.random.uniform(0, 1, sample):
            u = np.random.uniform(l, 1)
            l = np.min((l, 1))
            u = np.min((u, 1))
            if l == 0 and u == 1:
                data["rdm_type"].append("RDM")
            else:
                data["rdm_type"].append("RGTM")
            data["dataset"].append(dataset_name)
            data["l"].append(l)
            data["u"].append(u)
            data["p"].append(-1)
            data["k"].append(-1)
            data["rdm"].append(gt_transform(pdist(x), gt_type="lu", l=l, u=u))
            data["sorted_rdm"].append(
                gt_transform(
                    squareform(sort_dist(squareform(pdist(x)))), gt_type="lu", l=l, u=u
                )
            )
    return data


def gd_parameter_search(
    x,
    dataset_name="data",
    search_type="single",
    include_RDM=True,
    dictionary=None,
    **kwargs,
):
    if dictionary is None:
        data = {
            "rdm": [],
            "p": [],
            "k": [],
            "l": [],
            "u": [],
            "sorted_rdm": [],
            "dataset": [],
            "rdm_type": [],
        }
    else:
        data = dictionary
    N = x.shape[0]

    if include_RDM:
        # RDM
        data["rdm_type"].append("RDM")
        data["dataset"].append(dataset_name)
        data["p"].append(-1)
        data["k"].append(-1)
        data["l"].append(-1)
        data["u"].append(-1)
        data["rdm"].append(minmax_transform(rank_transform(pdist(x))))
        data["sorted_rdm"].append(
            squareform(
                sort_dist(squareform(minmax_transform(rank_transform(pdist(x)))))
            )
        )

    if search_type == "grid":
        ps = np.arange(0, 1 + 1e-4, 0.02)
        ks = np.arange(1, 21)
    elif search_type == "single":
        ps = [kwargs.get("p", 0.5)]
        ks = [kwargs.get("k", 5)]

    # RGDM - p-shortest
    for p in ps:
        data["rdm_type"].append(f"RGDM (p-shortest)")
        data["dataset"].append(dataset_name)
        data["p"].append(p)
        data["k"].append(-1)
        data["l"].append(-1)
        data["u"].append(-1)
        data["rdm"].append(geodesic_transform(pdist(x), gd_type="p_shortest", p=p))
        data["sorted_rdm"].append(
            geodesic_transform(
                squareform(sort_dist(squareform(pdist(x)))), gd_type="p_shortest", p=p
            )
        )

    # RGDM - p-nearest
    for p in ps:
        data["rdm_type"].append(f"RGDM (p-nearest)")
        data["dataset"].append(dataset_name)
        data["p"].append(p)
        data["k"].append(-1)
        data["l"].append(-1)
        data["u"].append(-1)
        data["rdm"].append(geodesic_transform(pdist(x), gd_type="p_nearest", p=p))
        data["sorted_rdm"].append(
            geodesic_transform(
                squareform(sort_dist(squareform(pdist(x)))), gd_type="p_nearest", p=p
            )
        )

    # RGDM - k-nearest
    for k in ks:
        data["rdm_type"].append(f"RGDM (k-nearest)")
        data["dataset"].append(dataset_name)
        data["p"].append(-1)
        data["k"].append(k)
        data["l"].append(-1)
        data["u"].append(-1)
        data["rdm"].append(geodesic_transform(pdist(x), gd_type="k_nearest", k=k))
        data["sorted_rdm"].append(
            geodesic_transform(
                squareform(sort_dist(squareform(pdist(x)))), gd_type="k_nearest", k=k
            )
        )

    return data


def gd_after_gt(x, l=0.0, u=0.075, dataset_name="data", dictionary=None, **kwargs):
    if dictionary is None:
        data = {
            "rdm": [],
            "p": [],
            "k": [],
            "l": [],
            "u": [],
            "sorted_rdm": [],
            "dataset": [],
            "rdm_type": [],
        }
    else:
        data = dictionary
    N = x.shape[0]
    data["rdm_type"].append(f"RGDM")
    data["dataset"].append(dataset_name)
    data["p"].append(-1)
    data["k"].append(-1)
    data["l"].append(l)
    data["u"].append(u)

    data["rdm"].append(
        geodesic_transform(
            gt_transform(pdist(x), gt_type="lu", l=l, u=u), gd_type="twostep"
        )
    )
    data["sorted_rdm"].append(
        geodesic_transform(
            gt_transform(
                squareform(sort_dist(squareform(pdist(x)))), gt_type="lu", l=l, u=u
            ),
            gd_type="twostep",
        )
    )

    return data


def map_df_labels(df):
    label_map = {}
    for p in topo_patterns:
        label_map[p] = "topology"
    for p in cate_patterns:
        label_map[p] = "clusters"
    for p in ring_patterns:
        label_map[p] = "rings"
    for p in eigt_patterns:
        label_map[p] = "eights"
    df["shape"] = df["dataset"].map(label_map)
    return df


def map_df_rings(df):
    label_map = {}
    for p in ["flat_ring", "twist_ring", "double_twist_ring"]:
        label_map[p] = "simple_rings"
    for p in ["two_rings", "mixed_rings", "locked_rings"]:
        label_map[p] = "complex_rings"
    df["ring_type"] = df["dataset"].map(label_map)
    return df


def rotate_matrix(m, a, center=None):
    R = np.array([[np.cos(-a), -np.sin(-a)], [np.sin(-a), np.cos(-a)]])
    if center is None:
        center = np.mean(m, axis=0)
    new_m = (m - center) @ R + center
    return new_m


def rotate_mds(pattern_mds):
    dx = np.mean(pattern_mds[[0, 2], 0]) - np.mean(pattern_mds[[1, 3], 0])
    dy = np.mean(pattern_mds[[0, 2], 1]) - np.mean(pattern_mds[[1, 3], 1])
    a = -np.arctan(dy / dx)
    new_pattern_mds = rotate_matrix(pattern_mds, a, center=None)
    return new_pattern_mds


def combine_dict_d(d1, d2):
    return {
        "distance": d1["distance"] + d2["distance"],
        "distance_type": d1["distance_type"] + d2["distance_type"],
        "subject": d1["subject"] + d2["subject"],
        "roi": d1["roi"] + d2["roi"],
        "length": d1["length"] + d2["length"],
    }


def rotate_matrix(mat):
    N = mat.shape[0]
    # Consider all squares one by one
    for x in range(0, int(N / 2)):
        # Consider elements in group
        # of 4 in current square
        for y in range(x, N - x - 1):
            # store current cell in temp variable
            temp = mat[x][y]

            # move values from right to top
            mat[x][y] = mat[y][N - 1 - x]

            # move values from bottom to right
            mat[y][N - 1 - x] = mat[N - 1 - x][N - 1 - y]

            # move values from left to bottom
            mat[N - 1 - x][N - 1 - y] = mat[N - 1 - y][x]

            # assign temp to left
            mat[N - 1 - y][x] = temp

    return mat


def check_lu(l, u):
    if u < l:
        l, u = u, l
    fail_mode = l + u <= 1e-2 or l + u >= 2 - 1e-2
    l, u = np.around(l, decimals=2), np.around(u, decimals=2) + 1e-5
    l, u = np.min((l, 1)), np.min((u, 1))
    return (not fail_mode), l, u


def check_gt(rtype, l, u):
    if rtype == "RGTM (local extractor)":
        metcondition = np.mean([l, u]) <= 0.2 + 1e-3
    elif rtype == "RGTM (global extractor)":
        metcondition = np.mean([l, u]) >= 0.8 - 1e-3
    elif rtype == "RGTM (geometry sensitive)":
        metcondition = np.abs(u - l) >= 0.8 - 1e-3
    elif rtype == "RGTM (topology sensitive)":
        metcondition = (l >= 0.4 - 1e-3) & (u <= 0.6 + 1e-3)
    elif rtype == "RGTM (intermediate)":
        metcondition = not (
            (l + u == 0)
            | (check_gt("RGTM (topology sensitive)", l, u))
            | (check_gt("RGTM (local extractor)", l, u))
            | (check_gt("RGTM (global extractor)", l, u))
            | (check_gt("RGTM (geometry sensitive)", l, u))
        )
    return metcondition


def run_rgtm_grid_search(
    roi_names,
    rdms,
    subjs,
    rois,
    dataset_name,
    mode,
    resolution,
    n_boots,
    boots_seq,
    print_progress,
    two_step=False,
    additional_stats=False,
):
    grid_size = len(np.arange(0, 1.0, resolution))

    mask = np.ones((grid_size + 1, grid_size + 1))
    mask[np.triu_indices_from(mask)] = False
    mask[-1, -1] = True
    mask[0, 0] = True

    if mode in ["load", "savefigure"]:
        acc_map_full = np.load(
            f"topology_figures/{dataset_name}_grid_search_acc_map.npy"
        )
    elif mode == "save":
        resolution = 0.05
        acc_map = np.zeros((grid_size, grid_size))

        for i, l in enumerate(np.arange(0, 1.0 - 1e-5, resolution)):
            for j, u in enumerate(np.arange(1 + 1e-4, l + 1e-5, -resolution)):
                if l + u <= 1e-2 or l + u >= 2 - 1e-2:
                    continue
                l, u = np.around(l, decimals=2), np.around(u, decimals=2) + 1e-5
                l, u = np.min((l, 1)), np.min((u, 1))
                if two_step:
                    transformed_rdms = np.array(
                        [
                            geodesic_transform(
                                gt_transform(rdm, l=l, u=u), gd_type="twostep"
                            )
                            for rdm in rdms
                        ]
                    )
                else:
                    transformed_rdms = np.array(
                        [gt_transform(rdm, l=l, u=u) for rdm in rdms]
                    )
                np.nan_to_num(transformed_rdms, posinf=finitemax(transformed_rdms))
                acc, _, _ = loo_eval(roi_names, transformed_rdms, subjs, rois)
                acc_map[i][j] = acc

        acc_map_full = np.zeros((grid_size + 1, grid_size + 1))
        acc_map_full[:grid_size, :grid_size] = acc_map

        np.save(
            f"topology_figures/{dataset_name}_grid_search_acc_map.npy", acc_map_full
        )

    return acc_map_full, mask


def run_and_plot_rdm_grid_search(
    roi_names,
    rdms,
    subjs,
    rois,
    dataset_name="trans62",
    heatmap_title="Brain Region Decodability",
    mode="save",
    resolution=0.05,
    n_boots=10,
    boots_seq=None,
    print_progress=False,
    procrustes_ref=None,
    legend=True,
    two_step=False,
):
    print("start grid search...")

    acc_map_full, mask = run_rgtm_grid_search(
        roi_names,
        rdms,
        subjs,
        rois,
        dataset_name,
        mode,
        resolution,
        n_boots,
        boots_seq,
        print_progress,
        two_step,
    )
    plot_grid_acc_map(acc_map_full, mask, mode, dataset_name)

    best_spots = np.where(acc_map_full == np.max(acc_map_full))
    acc_df = pd.DataFrame(
        acc_map_full.T,
        index=list(np.arange(1 + 1e-5, 0 + 1e-5, -resolution)) + [0],
        columns=list(np.arange(0, 1.0 - 1e-5, resolution)) + [1],
    )
    best_l, best_u = np.around(acc_df.columns[best_spots[0][0]], decimals=2), np.around(
        acc_df.index[best_spots[1][0]], decimals=2
    )
    print("best l and u:", best_l, best_u)

    dr = mds.fit_transform(rdms)
    if procrustes_ref is not None:
        procrustes_ref, dr, _ = procrustes(procrustes_ref, dr)
    else:
        procrustes_ref, dr, _ = procrustes(dr, dr)

    df = pd.DataFrame(
        {
            "subj": subjs.flatten(),
            "roi": [roi_names[r - 1] for r in rois.flatten()],
            "mds_0": dr[:, 0],
            "mds_1": dr[:, 1],
        }
    )

    plot_mds_dr(df, roi_names, mode, dataset_name, postfix="rdm", legend=legend)

    if two_step:
        transformed_rdms = np.array(
            [
                geodesic_transform(
                    gt_transform(rdm, l=best_l, u=best_u), gd_type="twostep"
                )
                for rdm in rdms
            ]
        )
    else:
        transformed_rdms = np.array(
            [gt_transform(rdm, l=best_l, u=best_u) for rdm in rdms]
        )

    dr = mds.fit_transform(transformed_rdms)
    procrustes_ref, dr, _ = procrustes(procrustes_ref, dr)
    df = pd.DataFrame(
        {
            "subj": subjs.flatten(),
            "roi": [roi_names[r - 1] for r in rois.flatten()],
            "mds_0": dr[:, 0],
            "mds_1": dr[:, 1],
        }
    )

    plot_mds_dr(df, roi_names, mode, dataset_name, postfix="rgtm", legend=legend)
    return procrustes_ref


def run_sample_rgtm(
    roi_names,
    rdms,
    subjs,
    rois,
    dataset_name="trans62",
    n_samples=10,
    n_boots=10,
    boots_seq=None,
    print_progress=False,
    mode="none",
    rdm=False,
    two_step=False,
):
    def get_gt_trios(rtype):
        notfound = True
        while notfound:
            l, u = np.random.uniform(size=2)
            found, l, u = check_lu(l, u)
            if found:
                notfound = not check_gt(rtype, l, u)
        return get_data_trios(l, u)

    def get_data_trios(l, u):
        if two_step:
            transformed_rdms = np.array(
                [
                    geodesic_transform(gt_transform(rdm, l=l, u=u), gd_type="twostep")
                    for rdm in rdms
                ]
            )
        else:
            transformed_rdms = np.array([gt_transform(rdm, l=l, u=u) for rdm in rdms])
        np.nan_to_num(transformed_rdms, posinf=finitemax(transformed_rdms))
        return transformed_rdms, subjs, rois

    if mode == "load":
        with open(f"topology_figures/{dataset_name}_unbiased_acc.pkl", "rb") as handle:
            plot_results_dict = pickle.load(handle)
    else:
        if rdm:
            lu_sampler = {
                "RDM": lambda: (rdms, subjs, rois),
                "RGTM (topology sensitive)": lambda: get_gt_trios(
                    "RGTM (topology sensitive)"
                ),
                "RGTM (geometry sensitive)": lambda: get_gt_trios(
                    "RGTM (geometry sensitive)"
                ),
                "RGTM (local extractor)": lambda: get_gt_trios(
                    "RGTM (local extractor)"
                ),
                "RGTM (global extractor)": lambda: get_gt_trios(
                    "RGTM (global extractor)"
                ),
                "RGTM (intermediate)": lambda: get_gt_trios("RGTM (intermediate)"),
            }
        else:
            lu_sampler = {
                "topology sensitive": lambda: get_gt_trios("RGTM (topology sensitive)"),
                "geometry sensitive": lambda: get_gt_trios("RGTM (geometry sensitive)"),
                "local extractor": lambda: get_gt_trios("RGTM (local extractor)"),
                "global extractor": lambda: get_gt_trios("RGTM (global extractor)"),
                "intermediate": lambda: get_gt_trios("RGTM (intermediate)"),
            }

        plot_results_dict = {}
        for rtype in lu_sampler.keys():
            accs, sample_boots_accs = [], []
            print(f"sampling for {rtype}")
            for i in range(n_samples):
                if print_progress:
                    print(rtype, i)
                success = False
                while not success:
                    try:
                        transformed_rdms, subjs, rois = lu_sampler[rtype]()
                        acc, _, _ = loo_eval(roi_names, transformed_rdms, subjs, rois)
                        success = True
                    except:
                        print("skipped due to fitting error.")
                boots_accs = get_boots_estimates(
                    roi_names,
                    transformed_rdms,
                    subjs,
                    rois,
                    n_boots=n_boots,
                    boots_seq=boots_seq,
                )
                accs.append(acc)  # N_SAMPLES
                sample_boots_accs.append(boots_accs)  # N_SAMPLES x N_BOOTS

            sample_boots_accs = np.array(sample_boots_accs)
            plot_results_dict[rtype] = {
                "sample_data_accs": accs,
                "mean_data_acc": np.nanmean(accs),
                "ustd": np.nanstd(np.nanmean(sample_boots_accs, axis=0)),
            }

            if mode == "save":
                f = open(f"topology_figures/{dataset_name}_unbiased_acc.pkl", "wb")
                pickle.dump(plot_results_dict, f)
                f.close()

    return plot_results_dict


def plot_patterns(patterns, use_mlab=True, savefig=False):
    for pattern in patterns:
        data, supp = get_shape(shape=pattern, sample=40)
        changed_supp = np.array([str(s) for s in supp]).copy()
        # changed_supp[[29,30,31,10,1,21]] = ''
        if use_mlab:
            fig, ax = plot3d(data, changed_supp, untouch=True, use_number=True)
        else:
            fig, ax = plot3d(data, changed_supp, untouch=True, use_number=True)
        if savefig:
            plt.savefig(f"topology_figures/pattern_{pattern}.png", dpi=300)


def plot_example_mds_rdm(
    selected_df,
    rdm_type,
    selected_patterns,
    selected_patterns_names,
    l=0,
    u=1,
    p=0.06,
    k=1,
    v1=False,
    save=True,
    rotate_mds_plot=True,
):
    if rdm_type == "RDM":
        small_df = selected_df[selected_df["rdm_type"] == rdm_type]
    elif rdm_type == "RGDM":
        small_df = selected_df[selected_df["rdm_type"] == rdm_type]
    elif rdm_type == "RGTM":
        small_df = selected_df[
            (selected_df["rdm_type"] == rdm_type)
            & np.isclose(selected_df["l"], l, atol=1e-2)
            & np.isclose(selected_df["u"], u, atol=1e-2)
        ]
    elif rdm_type == "RGDM (p-nearest)":
        small_df = selected_df[
            (selected_df["rdm_type"] == rdm_type)
            & np.isclose(selected_df["p"], p, atol=1e-2)
        ]
    elif rdm_type == "RGDM (k-nearest)":
        small_df = selected_df[
            (selected_df["rdm_type"] == rdm_type)
            & np.isclose(selected_df["k"], k, atol=1e-2)
        ]
    elif rdm_type == "RGDM (p-shortest)":
        small_df = selected_df[
            (selected_df["rdm_type"] == rdm_type)
            & np.isclose(selected_df["p"], p, atol=1e-2)
        ]

    pattern_rdms = []
    for p in selected_patterns:
        pattern_rdms.append(small_df[small_df["dataset"] == p].iloc[0]["rdm"])

    pattern_rdms = np.array(pattern_rdms)
    pattern_rdms = np.nan_to_num(pattern_rdms, posinf=finitemax(pattern_rdms))

    pattern_mds = mds.fit_transform(pattern_rdms)
    if rotate_mds_plot:
        pattern_mds = rotate_mds(pattern_mds)

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.plot(
        pattern_mds[[0, 2], 0],
        pattern_mds[[0, 2], 1],
        linestyle="-",
        label="geometrically similar",
        color="grey",
    )
    ax.plot(pattern_mds[[1, 3], 0], pattern_mds[[1, 3], 1], linestyle="-", color="grey")
    ax.plot(
        pattern_mds[[0, 1], 0],
        pattern_mds[[0, 1], 1],
        linestyle="dotted",
        label="topologically similar",
        color="grey",
    )
    ax.plot(
        pattern_mds[[2, 3], 0], pattern_mds[[2, 3], 1], linestyle="dotted", color="grey"
    )
    for i, p in enumerate(selected_patterns_names):
        ax.scatter(
            pattern_mds[i, 0],
            pattern_mds[i, 1],
            label=p,
            s=200,
            c="black" if "untangled" in p else "grey",
            marker="o" if "flat" in p else "X",
        )

    ax.axis("equal")
    plt.tick_params(
        left=False, right=False, labelleft=False, labelbottom=False, bottom=False
    )

    # plt.axis('off')
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    # plt.title(f'MDS of {rdm_type}')
    if save:
        plt.savefig(f"topology_figures/pattern_mds_{rdm_type}.png", dpi=300)

    fig, axes = plt.subplots(1, 4, figsize=(21, 5))
    for i, p in enumerate(selected_patterns_names):
        if v1:
            g = sns.heatmap(
                data=squareform(pattern_rdms[i]),
                vmin=0,
                # vmax=1,
                cmap="coolwarm",
                square=True,
                ax=axes[i],
            )
            pat = p.replace("_", " ")
            g.set_title(f"{rdm_type} of {pat}")
        else:
            g = sns.heatmap(
                data=squareform(pattern_rdms[i]),
                vmin=0,
                # vmax=1,
                cmap=color_scale(),
                linewidth=0.1,
                linecolor="black",
                cbar=False,
                yticklabels=False,
                xticklabels=False,
                square=True,
                ax=axes[i],
            )
            # pat = p.replace('_',' ')
            # g.set_title(f'{rdm_type} of {pat}')

    fig.tight_layout()
    filename = (
        rdm_type.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
    )
    if save:
        fig.savefig(f"topology_figures/pattern_rdm_{filename}.png", dpi=300)


def plot_rgtm_sensitivity(
    selected_df, selected_patterns, rdm_type="RGTM", resolution=0.05, remove_zero=True
):
    (
        x,
        y,
        between_touched,
        between_untouched,
        between_flat,
        between_twist,
        relative_ratio,
    ) = ([], [], [], [], [], [], [])
    for l in np.arange(0, 1.0, resolution):
        for u in np.arange(l, 1 + 1e-4, resolution):
            small_df = selected_df[
                np.isclose(selected_df["l"], l, atol=1e-2)
                & np.isclose(selected_df["u"], u, atol=1e-2)
            ]
            pattern_rdms = []
            for p in selected_patterns:
                pattern_rdms.append(small_df[small_df["dataset"] == p].iloc[0]["rdm"])
            pattern_rdms = np.array(pattern_rdms)
            pattern_rdms = np.nan_to_num(pattern_rdms, posinf=finitemax(pattern_rdms))
            pattern_mds = mds.fit_transform(pattern_rdms)
            if not remove_zero or l + u != 0:
                x.append(l)
                y.append(u)
                between_touched.append(np.linalg.norm(pattern_mds[0] - pattern_mds[1]))
                between_untouched.append(
                    np.linalg.norm(pattern_mds[2] - pattern_mds[3])
                )
                between_flat.append(np.linalg.norm(pattern_mds[0] - pattern_mds[2]))
                between_twist.append(np.linalg.norm(pattern_mds[1] - pattern_mds[3]))
                relative_ratio.append(
                    (
                        np.linalg.norm(pattern_mds[0] - pattern_mds[1])
                        + np.linalg.norm(pattern_mds[2] - pattern_mds[3])
                    )
                    / (
                        np.linalg.norm(pattern_mds[0] - pattern_mds[2])
                        + np.linalg.norm(pattern_mds[1] - pattern_mds[3])
                    )
                )

    min_between_touched = np.argmin(between_touched)
    min_between_untouched = np.argmin(between_untouched)
    max_between_flat = np.argmax(between_flat)
    max_between_twist = np.argmax(between_twist)
    min_relative_ratio = np.argmin(relative_ratio)

    print("min_between_touched", x[min_between_touched], y[min_between_touched])
    print("min_between_untouched", x[min_between_untouched], y[min_between_untouched])
    print("max_between_flat", x[max_between_flat], y[max_between_flat])
    print("max_between_twist", x[max_between_twist], y[max_between_twist])
    print("min_relative_ratio", x[min_relative_ratio], y[min_relative_ratio])

    fig, ax = plt.subplots(1, 5, figsize=(22, 4))

    ax[0].scatter(x, y, s=between_touched)
    ax[0].set_title(f"distances between touched")
    ax[1].scatter(x, y, s=between_untouched)
    ax[1].set_title(f"distances between untouched")
    ax[2].scatter(x, y, s=between_flat)
    ax[2].set_title(f"distances between flat")
    ax[3].scatter(x, y, s=between_twist)
    ax[3].set_title(f"distances between twist")
    ax[4].scatter(x, y, s=relative_ratio)
    ax[4].set_title(f"relative ratio")

    # plt.savefig(f'topology_figures/pattern_mds_{rdm_type}_detailed.png',dpi=300)


def plot_rgdm_sensitivity(
    selected_df,
    selected_patterns,
    rdm_type="RGDM (p-nearest)",
    ps=np.arange(0, 1 + 1e-4, 0.02),
    ks=np.arange(1, 21),
    remove_zero=True,
):
    (
        x,
        between_touched,
        between_untouched,
        between_flat,
        between_twist,
        relative_ratio,
    ) = ([], [], [], [], [], [])

    print(f"===== {rdm_type} =====")
    if rdm_type in ["RGDM (p-nearest)", "RGDM (p-shortest)"]:
        for p in ps:
            small_df = selected_df[np.isclose(selected_df["p"], p, atol=1e-2)]
            pattern_rdms = []
            for pat in selected_patterns:
                pattern_rdms.append(small_df[small_df["dataset"] == pat].iloc[0]["rdm"])
            pattern_rdms = np.array(pattern_rdms)
            pattern_rdms = np.nan_to_num(pattern_rdms, posinf=finitemax(pattern_rdms))
            pattern_mds = mds.fit_transform(pattern_rdms)
            if not remove_zero or p != 0:
                x.append(p)
                between_touched.append(np.linalg.norm(pattern_mds[0] - pattern_mds[1]))
                between_untouched.append(
                    np.linalg.norm(pattern_mds[2] - pattern_mds[3])
                )
                between_flat.append(np.linalg.norm(pattern_mds[0] - pattern_mds[2]))
                between_twist.append(np.linalg.norm(pattern_mds[1] - pattern_mds[3]))
                relative_ratio.append(
                    (
                        np.linalg.norm(pattern_mds[0] - pattern_mds[1])
                        + np.linalg.norm(pattern_mds[2] - pattern_mds[3])
                    )
                    / (
                        np.linalg.norm(pattern_mds[0] - pattern_mds[2])
                        + np.linalg.norm(pattern_mds[1] - pattern_mds[3])
                    )
                )
    elif rdm_type == "RGDM (k-nearest)":
        for k in ks:
            small_df = selected_df[np.isclose(selected_df["k"], k, atol=1e-2)]
            pattern_rdms = []
            for pat in selected_patterns:
                pattern_rdms.append(small_df[small_df["dataset"] == pat].iloc[0]["rdm"])
            pattern_rdms = np.array(pattern_rdms)
            pattern_rdms = np.nan_to_num(pattern_rdms, posinf=finitemax(pattern_rdms))
            pattern_mds = mds.fit_transform(pattern_rdms)
            if not remove_zero or k != 0:
                x.append(k)
                between_touched.append(np.linalg.norm(pattern_mds[0] - pattern_mds[1]))
                between_untouched.append(
                    np.linalg.norm(pattern_mds[2] - pattern_mds[3])
                )
                between_flat.append(np.linalg.norm(pattern_mds[0] - pattern_mds[2]))
                between_twist.append(np.linalg.norm(pattern_mds[1] - pattern_mds[3]))
                relative_ratio.append(
                    (
                        np.linalg.norm(pattern_mds[0] - pattern_mds[1])
                        + np.linalg.norm(pattern_mds[2] - pattern_mds[3])
                    )
                    / (
                        np.linalg.norm(pattern_mds[0] - pattern_mds[2])
                        + np.linalg.norm(pattern_mds[1] - pattern_mds[3])
                    )
                )

    min_between_touched = np.argmin(between_touched)
    min_between_untouched = np.argmin(between_untouched)
    max_between_flat = np.argmax(between_flat)
    max_between_twist = np.argmax(between_twist)
    min_relative_ratio = np.argmin(relative_ratio)

    print("min_between_touched", x[min_between_touched])
    print("min_between_untouched", x[min_between_untouched])
    print("max_between_flat", x[max_between_flat])
    print("max_between_twist", x[max_between_twist])
    print("min_relative_ratio", x[min_relative_ratio])

    fig, ax = plt.subplots(1, 5, figsize=(22, 4))

    ax[0].plot(x, between_touched)
    ax[0].set_title(f"distances between touched")
    ax[1].plot(x, between_touched)
    ax[1].set_title(f"distances between untouched")
    ax[2].plot(x, between_flat)
    ax[2].set_title(f"distances between flat")
    ax[3].plot(x, between_twist)
    ax[3].set_title(f"distances between twist")
    ax[4].plot(x, relative_ratio)
    ax[4].set_title(f"relative ratio")

    filename = (
        rdm_type.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
    )
    # plt.savefig(f'topology_figures/pattern_mds_{filename}_detailed.png',dpi=300)


def plot_grid_acc_map(acc_map_full, mask, mode, dataset_name):
    #     sns.set(font_scale = 1.5)
    #     sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 10))

    ax = sns.heatmap(
        matrix_ranktransform(acc_map_full.T),
        mask=rotate_matrix(mask),
        xticklabels=[],
        yticklabels=[],
        annot=False,
        annot_kws={"color": "black"},
        cmap="Greys",
        square=True,
        # vmin=0,vmax=1,
        cbar=False,
        # cbar_kws={"shrink": 0.5},
        linewidths=1,
        linecolor="black",
    )
    # ax.set_title(heatmap_title)
    # ax.set_xlabel('lower bound l')
    # ax.set_ylabel('upper bound u')
    plt.xticks(np.array([0, 1]) + 0.025)
    plt.yticks(np.array([0, 1]) + 0.025)

    best_ones = np.where(acc_map_full == np.max(acc_map_full))
    for i in np.arange(len(best_ones[0])):
        ax.add_patch(
            Rectangle(
                (best_ones[0][i], best_ones[1][i]),
                1,
                1,
                fill=False,
                edgecolor="red",
                lw=3,
            )
        )

    fig.tight_layout()
    if "save" in mode:
        plt.savefig(f"topology_figures/{dataset_name}_roi_acc_map.png", dpi=300)


def plot_mds_dr(
    df,
    roi_names,
    mode,
    dataset_name,
    postfix="rdm",
    legend=True,
    mdstitle="MDS of RDMs",
):
    #     sns.set(font_scale = 1.5)
    #     sns.set_style("white")
    colors = sns.color_palette("tab10", 10)

    fig, ax = plt.subplots(figsize=(10, 10))
    hue_parameters = {"data": df, "x": "mds_0", "y": "mds_1", "hue": "roi", "s": 200}
    g = sns.scatterplot(**hue_parameters, cmap=colors, ax=ax, legend=legend)
    #     g.set_title(mdstitle)
    #     g.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tick_params(
        left=False, right=False, labelleft=False, labelbottom=False, bottom=False
    )
    g.set(xlabel=None, ylabel=None)
    fig.tight_layout()

    if mode == "save":
        plt.savefig(f"topology_figures/{dataset_name}_mds_{postfix}.png", dpi=300)

    fig, ax = plt.subplots(figsize=(10, 10))

    hue_parameters = {"data": df, "x": "mds_0", "y": "mds_1", "hue": "roi", "s": 200}
    g = sns.scatterplot(**hue_parameters, cmap=colors, ax=ax, legend=legend)

    for i, roi in enumerate(roi_names):
        points = np.array(df[df["roi"] == roi][["mds_0", "mds_1"]])
        hull = ConvexHull(points)
        ax.fill(
            points[hull.vertices, 0], points[hull.vertices, 1], c=colors[i], alpha=0.2
        )

    #     g.set_title(mdstitle)
    #     g.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tick_params(
        left=False, right=False, labelleft=False, labelbottom=False, bottom=False
    )
    g.set(xlabel=None, ylabel=None)
    fig.tight_layout()
    if "save" in mode:
        plt.savefig(f"topology_figures/{dataset_name}_mds_{postfix}_hull.png", dpi=300)


def plot_sample_rgtm(
    plot_results_dict,
    dataset_name="trans62",
    pairs=None,
    title="Brain Region Decodability",
    ylim=1.5,
    mode="save",
    annotate_loc="inside",
    rdm=False,
    figsize=(8, 8),
):
    # colors = sns.color_palette()[:6]

    if mode in ["load", "savefigure"]:
        with open(f"topology_figures/{dataset_name}_unbiased_acc.pkl", "rb") as handle:
            plot_results_dict = pickle.load(handle)

    plt.rcParams["axes.grid"] = False

    rgtm_names = RGTM_NAMES_RDM if rdm else RGTM_NAMES
    x_pos = np.arange(len(rgtm_names))
    xs = [plot_results_dict[rn]["mean_data_acc"] for rn in rgtm_names]
    errs = [plot_results_dict[rn]["ustd"] for rn in rgtm_names]

    df_dict = {"mean_data_acc": [], "rdm_type": []}
    for rn in rgtm_names:
        df_dict["mean_data_acc"].append(plot_results_dict[rn]["mean_data_acc"])
        df_dict["rdm_type"].append(rn)

    sns.set_style("white")
    fig, ax = plt.subplots(figsize=figsize)

    #     g = sns.barplot(data=pd.DataFrame(data=df_dict), x='rdm_type', y='mean_data_acc', ax=ax, capsize=.2)
    #     g.axhline(y=df.groupby(['rgtm_type']).mean()['accuracy']['RDM'],ls='--',color='k')

    ax.bar(
        x_pos,
        xs,
        yerr=errs,
        align="center",
        alpha=0.5,
        color="grey",
        ecolor="black",
        capsize=10,
    )

    # ax.set_ylabel('accuracy')
    ax.set_xticks(x_pos)
    if rdm:
        ax.set_xticklabels(rgtm_names, rotation=10)
    else:
        ax.set_xticklabels(rgtm_names, rotation=0)
    # ax.set_title(title)
    ax.set_ylim([0, ylim])
    # ax.yaxis.grid(True)

    if pairs is None:
        pairs = list(itertools.combinations(rgtm_names, 2))

    if len(pairs) > 0:
        annotator = Annotator(
            ax,
            pairs,
            data=pd.DataFrame(df_dict),
            x="rdm_type",
            y="mean_data_acc",
            order=rgtm_names,
        )
        p_values = []
        for pair in pairs:
            ms = [plot_results_dict[pr]["mean_data_acc"] for pr in pair]
            ns = [len(plot_results_dict[pr]["sample_data_accs"]) for pr in pair]
            ss = [plot_results_dict[pr]["ustd"] for pr in pair]
            if 'trans62' in dataset_name:
                df = 24
            elif 'nn' in dataset_name:
                df = 10
            else:
                df = 'pooled'
            p_values.append(get_paired_p_values(ms, ns, ss, df))
        annotator.configure(
            text_format="star", loc=annotate_loc
        ).set_pvalues_and_annotate(p_values)

    plt.tick_params(right=False, labelbottom=False, bottom=False)

    plt.tight_layout()
    if "save" in mode:
        plt.savefig(f"topology_figures/{dataset_name}_unbiased_acc.png", dpi=300)
    if mode == "save":
        f = open(f"topology_figures/{dataset_name}_unbiased_acc.pkl", "wb")
        pickle.dump(plot_results_dict, f)
        f.close()

    plt.show()


def plot3d(
    data, supp=None, fig=None, ax=None, untouch=False, use_number=True, **kwargs
):
    sample = len(data)
    fig = fig if fig else plt.figure(figsize=(10, 10))
    ax = ax if ax else fig.add_subplot(111, projection="3d")
    #     ax.scatter(data[:int(sample/2), 0], data[:int(sample/2), 1], data[:int(sample/2), 2], s=200, c=supp[:int(sample/2)], alpha=1,cmap = 'Greys', **kwargs)
    ax.plot(data[:, 0], data[:, 1], data[:, 2], c="w", linewidth=10, **kwargs)
    ax.plot(
        data[int(sample / 2) - 1 :, 0],
        data[int(sample / 2) - 1 :, 1],
        data[int(sample / 2) - 1 :, 2],
        c="k",
        linewidth=5,
        **kwargs,
    )
    if untouch:
        ax.plot(
            data[: int(sample / 2), 0],
            data[: int(sample / 2), 1],
            data[: int(sample / 2), 2],
            c="w",
            linewidth=10,
            **kwargs,
        )
    ax.plot(
        data[: int(sample / 2), 0],
        data[: int(sample / 2), 1],
        data[: int(sample / 2), 2],
        c="k",
        linewidth=5,
        **kwargs,
    )
    ax.plot(
        data[[-1, 0, 1]][:, 0],
        data[[-1, 0, 1]][:, 1],
        data[[-1, 0, 1]][:, 2],
        c="w",
        linewidth=10,
        **kwargs,
    )
    ax.plot(
        data[[-2, -1, 0, 1, 2]][:, 0],
        data[[-2, -1, 0, 1, 2]][:, 1],
        data[[-2, -1, 0, 1, 2]][:, 2],
        c="k",
        linewidth=5,
        **kwargs,
    )
    #     ax.scatter(data[int(sample/2):, 0], data[int(sample/2):, 1], data[int(sample/2):, 2], s=200, c=supp[int(sample/2):], alpha=1,cmap = 'Greys', **kwargs)
    if use_number:
        s_size = 400
    else:
        s_size = 50
    ax.scatter(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        s=s_size,
        marker="o",
        color="white",
        edgecolors="grey",
        linewidth=2,
        alpha=1,
        **kwargs,
    )
    if use_number:
        for i, s in enumerate(supp):
            ax.text(data[i, 0] - 0.01, data[i, 1] - 0.1, data[i, 2], s, color="k")
    lim_range = 1.5
    ax.set_xlim([-lim_range, lim_range])
    ax.set_ylim([-lim_range, lim_range])
    ax.set_zlim([-lim_range, lim_range])
    return fig, ax

    # def plot3d_mlab(data, plot_type='line', colormap='Spectral', **kwargs):
    sample = len(data)
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    fig = mlab.figure(1, size=(400, 400), bgcolor=(1, 1, 1))
    mlab.clf()
    colors = (np.sin(np.arange(sample) * np.pi * 2 / sample) + 1) / 2
    colors1 = np.concatenate([colors, np.array([colors[0]])])
    if plot_type in ["line", "linepoint"]:
        x1, y1, z1 = (
            np.concatenate([x, np.array([x[0]])]),
            np.concatenate([y, np.array([y[0]])]),
            np.concatenate([z, np.array([z[0]])]),
        )
        mlab.plot3d(x1, y1, z1, colors1, tube_radius=0.1, colormap=colormap)
    if plot_type in ["point", "linepoint"]:
        nodes = mlab.points3d(x, y, z, scale_factor=0.25, colormap=colormap)
        nodes.glyph.scale_mode = "scale_by_vector"
        nodes.mlab_source.dataset.point_data.scalars = colors

    mlab.show()
    return fig, None


def color_scale(n_cols=1000, anchor_cols=None, monitor=False):
    """linearly interpolates between a set of given
    anchor colours to give n_cols and displays them
    if monitor is set

    Args:
        n_cols (int): number of colors for the colormap
        anchor_cols (numpy.ndarray, optional): what color space to
            interpolate. Defaults to None.
        monitor (boolean, optional): quick visualisation of the
            resulting colormap. Defaults to False.

    Returns:
        numpy.ndarray: n_cols x 3 RGB array.

    """

    if anchor_cols is None:
        # if no anchor_cols provided, use red to blue
        anchor_cols = np.array([[0, 0.4, 0.9], [0.8, 0.8, 0.8], [0.9, 0.3, 0.3]])
    # define color scale
    n_anchors = anchor_cols.shape[0]

    # simple 1D interpolation
    fn = interp1d(
        range(n_anchors),
        anchor_cols.T,
    )
    cols = fn(np.linspace(0, n_anchors - 1, n_cols)).T

    # optional visuals
    if monitor:
        reshaped_cols = cols.reshape((n_cols, 1, 3))
        width = int(n_cols / 2)
        mapping = np.tile(reshaped_cols, (width, 1))
        plt.imshow(mapping)
        plt.show()

    cmap = ListedColormap(cols)
    cmap.set_bad("white")

    return cmap
