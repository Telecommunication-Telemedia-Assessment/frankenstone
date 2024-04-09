#!/usr/bin/env python3
import os
import json
import argparse
import sys
import time
import concurrent.futures

import tensorflow as tf
import tensorflow_decision_forests as tfdf
tf.experimental.numpy.experimental_enable_numpy_behavior()
for x in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(x, True)

import pandas as pd


import dover_features
import fastvqa_features
import nvencc_features
import pxl_features
import vila_features
import musiq_features
import qalign_features

from utils import merge_dicts
from utils import timeit


def extract_features_video(video, frame_sampling):
    res = [fastvqa_features.extract_features(video)] # fastvqa must be performed in a single thread alone...
    features_fun = [
        nvencc_features.extract_features,
        dover_features.extract_features,
        pxl_features.extract_features,
        vila_features.extract_features,
        musiq_features.extract_features,
        qalign_features.extract_features
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(features_fun)) as executor:
        res.extend(executor.map(lambda x: x(video, frame_sampling), features_fun))

    return merge_dicts(res)


@timeit
def main(_):
    # argument parsing
    parser = argparse.ArgumentParser(description='frankenstone toolbox for UGC video quality models/features',
                                     epilog="stg7 2024",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", nargs="+", type=str, help="video to process")
    parser.add_argument("--frame_sampling", "-fs", action="store_true", help="use frame sampling")
    parser.add_argument("--features_folder", type=str, default="features", help="only for calculate features, folder to store the features")

    a = vars(parser.parse_args())

    for video in a["video"]:
        features = extract_features_video(video, a["frame_sampling"])

        print(json.dumps(features))
        featuresfile = os.path.join(
            a["features_folder"], os.path.splitext(os.path.basename(video))[0] + ".json"
        )
        os.makedirs(a["features_folder"], exist_ok=True)

        print(f"saving features in {featuresfile}")
        with open(featuresfile, "w") as xfp:
            json.dump(features, xfp, indent=4, sort_keys=True)




if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))



