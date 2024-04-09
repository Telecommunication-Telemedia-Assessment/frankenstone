#!/usr/bin/env python3

import os
import argparse
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 2 show only errors  #3 no output
os.environ["TFHUB_CACHE_DIR"] = os.path.join(os.path.dirname(__file__), "pretrained_weights/tfmodels")

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_models as tfm

tf.experimental.numpy.experimental_enable_numpy_behavior()
physical_devices = tf.config.list_physical_devices('GPU')
for x in physical_devices:
    tf.config.experimental.set_memory_growth(x, True)

import numpy as np
import pandas as pd

from utils import video_frames
from utils import sample_non_uniform
from utils import prefix_dict


def vila_model():
    # VILA model
    model_handle = 'https://tfhub.dev/google/vila/image/1'
    model = hub.load(model_handle)
    predict_fn = model.signatures['serving_default']
    output_key = "predictions"

    def predict(png_str):
        # VILA prediction
        res = predict_fn(png_str)
        return res[output_key]
    return predict


def extract_features(video_path, frame_sampling=True):
    # read video frames to gpu memory
    all_frames = video_frames(video_path)
    frames = sample_non_uniform(all_frames, k=6)  # k = 5
    print(f"vila features: process {len(frames)} of {len(all_frames)} frames")

    res = {}

    models_fun = {
        "vila": vila_model()
    }
    for model in models_fun:
        # estimate model predictions
        res[model] = tf.map_fn(
            fn=lambda x: models_fun[model](
                tf.io.encode_png(
                    tf.cast(
                        x,
                        tf.uint8
                    ),
                    compression=0
                ),
            ),
            elems=np.array(frames),
            fn_output_signature=tf.float32
        ).numpy().flatten()

        # estimate model cc predictions
        res[model + "_cc"] = tf.map_fn(
            fn=lambda x: models_fun[model](
                tf.io.encode_png(
                    tf.cast(
                        #tf.image.resize(
                            tf.image.central_crop(x, 0.5), #(272, 272),
                            #preserve_aspect_ratio=True, antialias=False
                        #),
                        tf.uint8
                    ),
                    compression=0
                ),
            ),
            elems=np.array(frames),
            fn_output_signature=tf.float32
        ).numpy().flatten()


    # pool the features
    df = pd.DataFrame(res)
    mean = prefix_dict(df.mean().to_dict(), "mean_")
    return mean


if __name__ == "__main__":

    # argument parsing
    parser = argparse.ArgumentParser(description='vila feature estimation',
                                     epilog="stg7 2024",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", type=str, nargs="+", help="video to extract scores")
    parser.add_argument("--features_folder", type=str, default="features_vila", help="only for calculate features, folder to store the features")

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


