#!/usr/bin/env python3

import argparse
import sys
import os
import gc
os.environ["TORCH_HOME"] = os.path.join(os.path.dirname(__file__), "pretrained_weights/tfmodels")
os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), "pretrained_weights/hugginface")


import json

import pyiqa
import torch
import torchvision
DEVICE = 'cuda'

import numpy as np
import pandas as pd
import decord


def video_frames(video_path, sampling=30):
    decord.bridge.set_bridge('torch')
    vr = decord.VideoReader(video_path)
    frames = []
    for x in range(0, len(vr), sampling):
        frame = vr[x]
        frames.append(vr[x])
    return frames


def prefix_dict(x, prefix):
    return {prefix + "" + k: x[k] for k in x}


def chunk_list(datas, chunksize):
    """Split list into the chucks

    Params:
        datas     (list): data that want to split into the chunk
        chunksize (int) : how much maximum data in each chunks

    Returns:
        chunks (obj): the chunk of list
    """

    for i in range(0, len(datas), chunksize):
        yield datas[i:i + chunksize]


def extract_features(video_path):
    musiq_model = pyiqa.create_metric('musiq').cuda()

    print(f"musiq features of {video_path}")
    values = []
    for frame_batch in chunk_list(video_frames(video_path), 2):
        #musiq_scores = musiq_model(
        #    torch.cat([x.permute(2,0, 1).unsqueeze(0) for x in frame_batch])
        #)
        #values.extend(musiq_scores.cpu().numpy().flatten())
        musiq_scores = musiq_model(
            torch.cat([torchvision.transforms.CenterCrop(size=(2*224, 2*224))(x.permute(2,0, 1)).unsqueeze(0) for x in frame_batch])
        )
        values.extend(musiq_scores.cpu().numpy().flatten())

    res = {
        "musiq": values,
        "musiq_cc": values
    }

    df = pd.DataFrame(res)
    mean = prefix_dict(df.mean().to_dict(), "mean_")
    last_first_diff = prefix_dict((df.iloc[-1] - df.iloc[0]).to_dict(), "last_first_diff_")
    return dict(mean, **last_first_diff)


if __name__ == "__main__":

    # argument parsing
    parser = argparse.ArgumentParser(description='musiq feature estimation',
                                     epilog="stg7 2024",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", type=str, nargs="+", help="video to extract scores")
    parser.add_argument("--features_folder", type=str, default="features_musiq", help="only for calculate features, folder to store the features")

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

