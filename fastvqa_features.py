#!/usr/bin/env python3
import torch
import yaml
import decord
from fastvqa.datasets import get_spatial_fragments, SampleFrames, FragmentSampleFrames
from fastvqa.models import DiViDeAddEvaluator
import numpy as np
import os
import json
import argparse


def sigmoid_rescale(score, model="FasterVQA"):
    mean, std = mean_stds[model]
    x = (score - mean) / std
    print(f"Inferring with model [{model}]:")
    score = 1 / (1 + np.exp(-x))
    return score


mean_stds = {
    "FasterVQA": (0.14759505, 0.03613452),
    "FasterVQA-MS": (0.15218826, 0.03230298),
    "FasterVQA-MT": (0.14699507, 0.036453716),
    "FAST-VQA":  (-0.110198185, 0.04178565),
    "FAST-VQA-M": (0.023889644, 0.030781006),
}

opts = {
    "FasterVQA": os.path.join(os.path.dirname(__file__), "f3dvqa-b.yml"),
    "FasterVQA-MS": os.path.join(os.path.dirname(__file__),"fastervqa-ms.yml"),
    "FasterVQA-MT": os.path.join(os.path.dirname(__file__),"fastervqa-mt.yml"),
    "FAST-VQA": os.path.join(os.path.dirname(__file__),"fast-b.yml"),
    "FAST-VQA-M": os.path.join(os.path.dirname(__file__),"fast-m.yml"),
}


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def extract_features(video, frame_sampling=True):
    args = {
        "model": "FasterVQA",
        "device": "cuda",
        "video_path": video
    }

    args = dotdict(args)

    video_reader = decord.VideoReader(args.video_path)

    opt = opts.get(args.model, opts["FAST-VQA"])
    with open(opt, "r") as f:
        opt = yaml.safe_load(f)

    ### Model Definition

    evaluator = DiViDeAddEvaluator(**opt["model"]["args"]).to(args.device)
    evaluator.load_state_dict(torch.load(opt["test_load_path"], map_location=args.device)["state_dict"])

    ### Data Definition
    vsamples = {}
    t_data_opt = opt["data"]["val-kv1k"]["args"]
    s_data_opt = opt["data"]["val-kv1k"]["args"]["sample_types"]
    for sample_type, sample_args in s_data_opt.items():
        ## Sample Temporally
        if t_data_opt.get("t_frag",1) > 1:
            sampler = FragmentSampleFrames(fsize_t=sample_args["clip_len"] // sample_args.get("t_frag",1),
                                           fragments_t=sample_args.get("t_frag",1),
                                           num_clips=sample_args.get("num_clips",1),
                                          )
        else:
            sampler = SampleFrames(clip_len = sample_args["clip_len"], num_clips = sample_args["num_clips"])

        num_clips = sample_args.get("num_clips",1)
        frames = sampler(len(video_reader))
        print("Sampled frames are", frames)
        frame_dict = {idx: video_reader[idx] for idx in np.unique(frames)}
        imgs = [frame_dict[idx] for idx in frames]
        video = torch.stack(imgs, 0)
        video = video.permute(3, 0, 1, 2)

        ## Sample Spatially
        sampled_video = get_spatial_fragments(video, **sample_args)
        mean, std = torch.FloatTensor([123.675, 116.28, 103.53]), torch.FloatTensor([58.395, 57.12, 57.375])
        sampled_video = ((sampled_video.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)

        sampled_video = sampled_video.reshape(sampled_video.shape[0], num_clips, -1, *sampled_video.shape[2:]).transpose(0,1)
        vsamples[sample_type] = sampled_video.to(args.device)
        print(sampled_video.shape)
    result = evaluator(vsamples)
    score = float(sigmoid_rescale(result.mean().item(), model=args.model))
    raw = float(result.mean().item())

    return {
        "fastervqa_score": score,
        "fastervqa_raw": raw
    }


if __name__ == "__main__":

    # argument parsing
    parser = argparse.ArgumentParser(description='fastervqa features estimation',
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