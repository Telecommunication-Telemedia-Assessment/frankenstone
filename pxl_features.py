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

import decord
from decord import VideoReader


from utils import weighted_mean
from utils import video_frames
from utils import sample_non_uniform
from utils import prefix_dict


def create_tf_lite_nima_model(MODELPATH="aesthetic_model.tflite"):
    # the models are taken from https://github.com/SophieMBerger/TensorFlow-Lite-implementation-of-Google-NIMA/tree/master

    interpreter = tf.lite.Interpreter(model_path=MODELPATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    def predict(image):
        image = tf.cast(
            tf.image.resize(image, size=tuple(input_details[0]['shape'][1:3])),
            input_details[0]["dtype"]
        )
        ff = tf.expand_dims(image, 0)

        interpreter.set_tensor(input_details[0]['index'], ff)
        interpreter.invoke()
        res = []
        for x in range(len(output_details)):
            output_data = interpreter.get_tensor(output_details[x]['index'])
            res.append(output_data)
        return weighted_mean(res[0].flatten())
    return predict


def calc_si(image):

    #  TODO: SI is usually performed on L channel
    image = tf.cast(image, tf.float32)
    image = tf.expand_dims(image, 0)
    sobel = tf.image.sobel_edges(image)
    #sobel_y = np.asarray(sobel[0, :, :, :, 0])
    #sobel_x = np.asarray(sobel[0, :, :, :, 1])
    #si = np.hypot(sobel_x, sobel_y).std()
    si = tf.experimental.numpy.std(
        tf.experimental.numpy.hypot(sobel[0, :, :, :, 0], sobel[0, :, :, :, 1])
    )
    return si.numpy()


def calc_colorfulness(image_rgb):
    rg = (image_rgb[:, :, 0] - image_rgb[:, :, 1]).ravel()
    yb = (image_rgb[:, :, 0] / 2 + image_rgb[:, :, 1] / 2 - image_rgb[:, :, 2]).ravel()

    rg_std = tf.experimental.numpy.std(rg)
    yb_std = tf.experimental.numpy.std(yb)

    rg_mean = tf.experimental.numpy.mean(rg)
    yb_mean = tf.experimental.numpy.mean(yb)

    trigo_len_std = tf.experimental.numpy.sqrt(rg_std ** 2 + yb_std ** 2)
    neutral_dist = tf.experimental.numpy.sqrt(rg_mean ** 2 + yb_mean ** 2)

    return (trigo_len_std + 0.3 * neutral_dist).numpy()


def calc_avg_luminance(f):
    return tf.experimental.numpy.mean(
        tf.experimental.numpy.mean(f, axis=2)
    ).numpy()


def calc_sharpness(image):
    img = tf.cast(image, tf.float32)
    blurry = tfm.vision.augment.gaussian_filter2d(
        img,
        filter_shape=(3,3),
        sigma=2,
        padding="REFLECT"
    )
    return tf.keras.metrics.mean_squared_error(blurry, img).numpy().mean()


def calc_ti(prev_frame, curr_frame):
    if prev_frame is None or curr_frame is None:
        return 0
    ti = tf.experimental.numpy.std(curr_frame - prev_frame)
    return ti.numpy()


def calc_ti_pair(pair):
    return calc_ti(pair[0], pair[1])


def calc_ssim_pair(pair):
    return tf.image.ssim(pair[0], pair[1], max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)


def extract_features(video_path):
    # read video frames to gpu memory
    all_frames = video_frames(video_path)
    frames = sample_non_uniform(all_frames, k=6)  # k = 5
    print(f"pxl features: process {len(frames)} of {len(all_frames)} frames")

    res = {}

    # signal based features and nima
    features_fun = {
        "nima_a": create_tf_lite_nima_model(os.path.join(os.path.dirname(__file__), "pretrained_weights/aesthetic_model.tflite")),
        "nima_q": create_tf_lite_nima_model(os.path.join(os.path.dirname(__file__), "pretrained_weights/technical_model.tflite")),
        "si": calc_si,
        "colorfulness": calc_colorfulness,
        "avg_luminance": calc_avg_luminance,
        "sharpness": calc_sharpness
    }

    for feature in features_fun:
        res[feature] = tf.map_fn(
            fn=features_fun[feature],
            elems=np.array(frames),
            fn_output_signature=tf.float32,
        ).numpy().flatten()
        res[feature + "_cc"] = tf.map_fn(
            fn=lambda x: features_fun[feature](tf.image.central_crop(x, 0.5)),
            elems=np.array(frames),
            fn_output_signature=tf.float32,
        ).numpy().flatten()

    # ti is a special case because of frame pairs
    res["ti"] = tf.map_fn(
        fn=calc_ti_pair,
        elems=np.array([(frames[0], frames[0])] + list(zip(frames[0:-1], frames[1:]))),
        fn_output_signature=tf.float32,
        #fn_output_signature=tf.string,
    ).numpy().flatten()

    res["ssim_pair"] = tf.map_fn(
        fn=calc_ssim_pair,
        elems=np.array([(frames[0], frames[0])] + list(zip(frames[0:-1], frames[1:]))),
        fn_output_signature=tf.float32,
        #fn_output_signature=tf.string,
    ).numpy().flatten()

    # ti to the first frame in the list
    res["ti_first"] = tf.map_fn(
        fn=calc_ti_pair,
        elems=np.array([(frames[0], frames[0])] + list(zip(frames[0:-1], [frames[0] for _ in range(len(frames) - 1)]))),
        fn_output_signature=tf.float32,
    ).numpy().flatten()

    res["ssim_pair_first"] = tf.map_fn(
        fn=calc_ssim_pair,
        elems=np.array([(frames[0], frames[0])] + list(zip(frames[0:-1], [frames[0] for _ in range(len(frames) - 1)]))),
        fn_output_signature=tf.float32,
    ).numpy().flatten()

    # pool the features
    df = pd.DataFrame(res)
    mean = prefix_dict(df.mean().to_dict(), "mean_")
    #breakpoint()

    return mean


if __name__ == "__main__":

    # argument parsing
    parser = argparse.ArgumentParser(description='pixel features estimation',
                                     epilog="stg7 2024",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", type=str, nargs="+", help="video to extract scores")
    parser.add_argument("--features_folder", type=str, default="features_pxl", help="only for calculate features, folder to store the features")

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

