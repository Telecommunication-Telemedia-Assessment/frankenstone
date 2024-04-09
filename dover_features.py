#!/usr/bin/env python3
import os
import argparse
import json
os.environ["TORCH_HOME"] = os.path.join(os.path.dirname(__file__), "pretrained_weights/torch")
os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), "pretrained_weights/hugginface")

import torch

import decord
import numpy as np
#import yaml

from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition
from dover.models import DOVER

mean, std = (
    torch.FloatTensor([123.675, 116.28, 103.53]),
    torch.FloatTensor([58.395, 57.12, 57.375]),
)


def fuse_results(results: list):
    x = (results[0] - 0.1107) / 0.07355 * 0.6104 + (
        results[1] + 0.08285
    ) / 0.03774 * 0.3896
    return 1 / (1 + np.exp(-x))


def gaussian_rescale(pr):
    # The results should follow N(0,1)
    pr = (pr - np.mean(pr)) / np.std(pr)
    return pr


def uniform_rescale(pr):
    # The result scores should follow U(0,1)
    return np.arange(len(pr))[np.argsort(pr).argsort()] / len(pr)



class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def extract_features(video):

    args = {
        #"opt": os.path.join(os.path.dirname(__file__), "./dover.yml"), # "the option file"
        "video_path": video, # the video to be processed
        "device": "cuda",
    }
    args = dotdict(args)

    # with open(args.opt, "r") as f:
    #     opt = yaml.safe_load(f)
    # print(opt)
    opt = {'name': 'DOVER', 'num_epochs': 0, 'l_num_epochs': 10, 'warmup_epochs': 2.5, 'ema': True, 'save_model': True, 'batch_size': 8, 'num_workers': 6, 'split_seed': 42, 'wandb': {'project_name': 'DOVER'}, 'data': {'val-livevqc': {'type': 'ViewDecompositionDataset', 'args': {'weight': 0.598, 'phase': 'test', 'anno_file': './examplar_data_labels/LIVE_VQC/labels.txt', 'data_prefix': '../datasets/LIVE_VQC/', 'sample_types': {'technical': {'fragments_h': 7, 'fragments_w': 7, 'fsize_h': 32, 'fsize_w': 32, 'aligned': 32, 'clip_len': 32, 'frame_interval': 2, 'num_clips': 3}, 'aesthetic': {'size_h': 224, 'size_w': 224, 'clip_len': 32, 'frame_interval': 2, 't_frag': 32, 'num_clips': 1}}}}, 'val-kv1k': {'type': 'ViewDecompositionDataset', 'args': {'weight': 0.54, 'phase': 'test', 'anno_file': './examplar_data_labels/KoNViD/labels.txt', 'data_prefix': '../datasets/KoNViD/', 'sample_types': {'technical': {'fragments_h': 7, 'fragments_w': 7, 'fsize_h': 32, 'fsize_w': 32, 'aligned': 32, 'clip_len': 32, 'frame_interval': 2, 'num_clips': 3}, 'aesthetic': {'size_h': 224, 'size_w': 224, 'clip_len': 32, 'frame_interval': 2, 't_frag': 32, 'num_clips': 1}}}}, 'val-ltest': {'type': 'ViewDecompositionDataset', 'args': {'weight': 0.603, 'phase': 'test', 'anno_file': './examplar_data_labels/LSVQ/labels_test.txt', 'data_prefix': '../datasets/LSVQ/', 'sample_types': {'technical': {'fragments_h': 7, 'fragments_w': 7, 'fsize_h': 32, 'fsize_w': 32, 'aligned': 32, 'clip_len': 32, 'frame_interval': 2, 'num_clips': 3}, 'aesthetic': {'size_h': 224, 'size_w': 224, 'clip_len': 32, 'frame_interval': 2, 't_frag': 32, 'num_clips': 1}}}}, 'val-l1080p': {'type': 'ViewDecompositionDataset', 'args': {'weight': 0.62, 'phase': 'test', 'anno_file': './examplar_data_labels/LSVQ/labels_1080p.txt', 'data_prefix': '../datasets/LSVQ/', 'sample_types': {'technical': {'fragments_h': 7, 'fragments_w': 7, 'fsize_h': 32, 'fsize_w': 32, 'aligned': 32, 'clip_len': 32, 'frame_interval': 2, 'num_clips': 3}, 'aesthetic': {'size_h': 224, 'size_w': 224, 'clip_len': 32, 'frame_interval': 2, 't_frag': 32, 'num_clips': 1}}}}, 'val-cvd2014': {'type': 'ViewDecompositionDataset', 'args': {'weight': 0.576, 'phase': 'test', 'anno_file': './examplar_data_labels/CVD2014/labels.txt', 'data_prefix': '../datasets/CVD2014/', 'sample_types': {'technical': {'fragments_h': 7, 'fragments_w': 7, 'fsize_h': 32, 'fsize_w': 32, 'aligned': 32, 'clip_len': 32, 'frame_interval': 2, 'num_clips': 3}, 'aesthetic': {'size_h': 224, 'size_w': 224, 'clip_len': 32, 'frame_interval': 2, 't_frag': 32, 'num_clips': 1}}}}, 'val-ytugc': {'type': 'ViewDecompositionDataset', 'args': {'weight': 0.443, 'phase': 'test', 'anno_file': './examplar_data_labels/YouTubeUGC/labels.txt', 'data_prefix': '../datasets/YouTubeUGC/', 'sample_types': {'technical': {'fragments_h': 7, 'fragments_w': 7, 'fsize_h': 32, 'fsize_w': 32, 'aligned': 32, 'clip_len': 32, 'frame_interval': 2, 'num_clips': 3}, 'aesthetic': {'size_h': 224, 'size_w': 224, 'clip_len': 32, 'frame_interval': 2, 't_frag': 32, 'num_clips': 1}}}}}, 'model': {'type': 'DOVER', 'args': {'backbone': {'technical': {'type': 'swin_tiny_grpb', 'checkpoint': True, 'pretrained': None}, 'aesthetic': {'type': 'conv_tiny'}}, 'backbone_preserve_keys': 'technical,aesthetic', 'divide_head': True, 'vqa_head': {'in_channels': 768, 'hidden_channels': 64}}}, 'optimizer': {'lr': 0.001, 'backbone_lr_mult': 0.1, 'wd': 0.05}, 'test_load_path': './pretrained_weights/DOVER.pth'}

    ### Load DOVER
    evaluator = DOVER(**opt["model"]["args"]).to(args.device)
    evaluator.load_state_dict(
        torch.load(opt["test_load_path"], map_location=args.device)
    )

    dopt = opt["data"]["val-l1080p"]["args"]

    temporal_samplers = {}
    for stype, sopt in dopt["sample_types"].items():
        if "t_frag" not in sopt:
            # resized temporal sampling for TQE in DOVER
            temporal_samplers[stype] = UnifiedFrameSampler(
                sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
            )
        else:
            # temporal sampling for AQE in DOVER
            temporal_samplers[stype] = UnifiedFrameSampler(
                sopt["clip_len"] // sopt["t_frag"],
                sopt["t_frag"],
                sopt["frame_interval"],
                sopt["num_clips"],
            )

    ### View Decomposition
    views, _ = spatial_temporal_view_decomposition(
        args.video_path, dopt["sample_types"], temporal_samplers
    )

    for k, v in views.items():
        num_clips = dopt["sample_types"][k].get("num_clips", 1)
        views[k] = (
            ((v.permute(1, 2, 3, 0) - mean) / std)
            .permute(3, 0, 1, 2)
            .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
            .transpose(0, 1)
            .to(args.device)
        )


    results = [r.mean().item() for r in evaluator(views)]

    # predict fused overall score, with default score-level fusion parameters
    res = {
        "dover": fuse_results(results),
        "dover_res_0": results[0],
        "dover_res_1": results[1],
    }
    return res


if __name__ == "__main__":

    # argument parsing
    parser = argparse.ArgumentParser(description='dover features estimation',
                                     epilog="stg7 2024",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", type=str, nargs="+", help="video to extract scores")
    parser.add_argument("--features_folder", type=str, default="features_dover", help="only for calculate features, folder to store the features")

    a = vars(parser.parse_args())
    for video in a["video"]:
        features = extract_features(video)
        features["video"] = video
        print(features)
        featuresfile = os.path.join(
            a["features_folder"], os.path.splitext(os.path.basename(video))[0] + ".json"
        )
        os.makedirs(a["features_folder"], exist_ok=True)

        print(f"saving features in {featuresfile}")
        with open(featuresfile, "w") as xfp:
            json.dump(features, xfp, indent=4, sort_keys=True)

