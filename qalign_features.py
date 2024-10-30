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

import numpy as np
import pandas as pd
import decord


from utils import video_frames
from utils import sample_non_uniform
from utils import prefix_dict


def extract_features(video_path, frame_sampling=True):
    device = torch.device("cuda") # torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    qalign_model = pyiqa.create_metric('qalign', device=device)

    print(f"qalign features of {video_path}")
    values = []
    for frame in video_frames(video_path, bridge="torch"):
        qalign_quality = qalign_model(
           frame.permute(2,0, 1).unsqueeze(0) / 255.0,
           task_='quality'
        )
        qalign_aesthetic = qalign_model(
           frame.permute(2,0, 1).unsqueeze(0)/ 255.0,
           task_='aesthetic'
        )
        # do cc calculations
        frame_cc = torchvision.transforms.CenterCrop(size=(2*224, 2*224))(frame.permute(2,0, 1)).unsqueeze(0) / 255.0
        qalign_quality_cc = qalign_model(
           frame_cc,
           task_='quality'
        )
        qalign_aesthetic_cc = qalign_model(
           frame_cc,
           task_='aesthetic'
        )

        values.append(
            {
                "qalign_quality": qalign_quality.cpu().numpy()[0],
                "qalign_aesthetic": qalign_aesthetic.cpu().numpy()[0],
                "qalign_quality_cc": qalign_quality_cc.cpu().numpy()[0],
                "qalign_aesthetic_cc": qalign_aesthetic_cc.cpu().numpy()[0],
            }
        )

    df = pd.DataFrame(values)
    mean = prefix_dict(df.mean().to_dict(), "mean_")
    return mean


if __name__ == "__main__":

    # argument parsing
    parser = argparse.ArgumentParser(description='q align feature estimation',
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

