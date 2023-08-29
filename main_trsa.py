#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# topological representational similarity analysis
# author: baihan lin (doerlbh@gmail.com)


from analysis_utils import *


def four_pattern_example():
    selected_patterns = ["flat_8", "twist_8", "untouched_flat_8", "untouched_twist_8"]
    selected_patterns_names = [
        "flat 8",
        "twisted 8",
        "untangled flat 8",
        "untangled twisted 8",
    ]
    data, supp = get_shape(shape=selected_patterns[0], sample=50)

    datasets_dict = get_all_patterns(patterns=selected_patterns, sample=50)
    selected_df, data = get_dataset_df(
        datasets_dict=datasets_dict,
        resolution=0.025,
        resolution_max=0.25,
        l=0.0,
        u=0.075,
        gt_search_type="grid",
        gd_search_type="grid",
        gt_type="all",
    )

    plot_example_mds_rdm(selected_df, "RDM", selected_patterns, selected_patterns_names)
    plot_example_mds_rdm(
        selected_df, "RGTM", selected_patterns, selected_patterns_names, l=0.0, u=0.075
    )
    plot_example_mds_rdm(
        selected_df, "RGDM", selected_patterns, selected_patterns_names
    )


def human_fmri_example():
    roi_names, rdms, subjs, rois = get_dataset(dataset_name="trans62", env="mac")
    n_sess, n_subjs, n_stims, n_rois = get_dataset_configs(rdms, subjs, rois)

    n_boots = 1000
    boots_seq = get_boots_seq(
        n_boots=n_boots, n_subjs=n_subjs, n_stims=n_stims, n_sess=n_sess
    )

    plot_results_dict = run_sample_rgtm(
        roi_names,
        rdms,
        subjs,
        rois,
        dataset_name="trans62",
        n_samples=10,
        n_boots=n_boots,
        boots_seq=boots_seq,
        mode="load",
    )
    plot_sample_rgtm(
        plot_results_dict,
        dataset_name="trans62",
        pairs=None,
        title="Brain Region Decodability",
        mode="load",
    )

    _ = run_and_plot_rdm_grid_search(
        roi_names,
        rdms,
        subjs,
        rois,
        dataset_name="trans62",
        mode="load",
        n_boots=n_boots,
        boots_seq=boots_seq,
    )


def dnn_noise_example():
    roi_names, rdms, subjs, rois = get_dataset(dataset_name=f"cnn62_g_0p0", env="mac")

    n_sess, n_subjs, n_stims, n_rois = get_dataset_configs(rdms, subjs, rois)
    print(n_sess, n_subjs, n_stims, n_rois)

    n_boots = 1000
    boots_seq = get_boots_seq(
        n_boots=n_boots, n_subjs=n_subjs, n_stims=n_stims, n_sess=n_sess
    )

    plot_results_dicts = []

    for noise_level in np.arange(5):
        print(f"========== noise level: {noise_level} ============")
        try:
            roi_names, rdms, subjs, rois = get_dataset(
                dataset_name=f"cnn62_g_0p{noise_level}", env="mac"
            )
            plot_results_dict = run_sample_rgtm(
                roi_names,
                rdms,
                subjs,
                rois,
                mode="load",
                dataset_name=f"cnn62_g_0p{noise_level}",
                n_samples=10,
                n_boots=n_boots,
                boots_seq=boots_seq,
            )
            plot_results_dicts.append(plot_results_dict)
            plot_sample_rgtm(
                plot_results_dict,
                dataset_name=f"cnn62_g_0p{noise_level}",
                pairs=None,
                title="Neural Net Layer Decodability",
                mode="load",
            )
        except:
            print(f"analysis stopped at noise level {noise_level}.")

    procrustes_ref = None
    for noise_level in np.arange(5):
        print(f"========== noise level: {noise_level} ============")
        try:
            roi_names, rdms, subjs, rois = get_dataset(
                dataset_name=f"cnn62_g_0p{noise_level}", env="mac"
            )
            procrustes_ref = run_and_plot_rdm_grid_search(
                roi_names,
                rdms,
                subjs,
                rois,
                dataset_name=f"cnn62_g_0p{noise_level}",
                heatmap_title="Neural Net Layer Decodability",
                mode="load",
                procrustes_ref=procrustes_ref,
                legend=False,
                n_boots=n_boots,
                boots_seq=boots_seq,
            )
        except:
            print(f"analysis stopped at noise level {noise_level}.")
