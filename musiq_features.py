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


from utils import video_frames
from utils import sample_non_uniform
from utils import prefix_dict
from utils import chunk_list


def extract_features(video_path, frame_sampling=True):
    musiq_model = pyiqa.create_metric('musiq').cuda()

    print(f"musiq features of {video_path}")
    values = []
    values_cc = []
    for frame_batch in chunk_list(video_frames(video_path, bridge="torch"), 2):
        musiq_scores = musiq_model(
           torch.cat([x.permute(2,0, 1).unsqueeze(0) for x in frame_batch])
        )
        values.extend(musiq_scores.cpu().numpy().flatten())
        musiq_scores_cc = musiq_model(
            torch.cat([torchvision.transforms.CenterCrop(size=(2*224, 2*224))(x.permute(2,0, 1)).unsqueeze(0) for x in frame_batch])
        )
        values_cc.extend(musiq_scores_cc.cpu().numpy().flatten())

    res = {
        "musiq": values,
        "musiq_cc": values_cc
    }

    df = pd.DataFrame(res)
    mean = prefix_dict(df.mean().to_dict(), "mean_")
    return mean


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

