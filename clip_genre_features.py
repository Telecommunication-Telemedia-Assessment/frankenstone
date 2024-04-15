#!/usr/bin/env python3
import argparse
import os
import json

import torch
import torchvision
from PIL import Image
import open_clip
import numpy as np
import pandas as pd

from utils import video_frames
from utils import prefix_dict


def extract_features(video_path, frame_sampling=True):
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k',
        cache_dir=os.path.join(os.path.dirname(__file__),"pretrained_weights/torch"),
        #device="cuda"
    )
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    genres = [
        "animated photo",
        "music photo",
        "computer game photo"
        "lecture photo",
        "text photo",
        "news photo",
        "sports photo",
        "vlog photo",
        "vr photo",
        "blocky photo",
        "unsharp photo",
        "animal photo",
        "portrait photo",
        "people photo",
        "colorful photo",
        "artistic photo"
    ] # these are not all genres
    text = tokenizer(genres)

    values = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text)
        for frame in video_frames(video_path, bridge="torch"):
            image_features = model.encode_image(
                preprocess(
                    Image.fromarray(
                        torchvision.transforms.Resize(size=(2*224, 2*224), antialias=False)(
                                frame.permute(2,0, 1)
                        ).permute(1,2, 0).cpu().numpy()
                    )
                ).unsqueeze(0)
            )
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()
            # find best matching genre
            v = {
                "clip_genre": genres[np.argmax(text_probs)].split(" ")[0]
            }
            values.append(v)
    genres_sorted = pd.DataFrame(values).value_counts()
    res = {
        "clip_genre_max": genres_sorted.index[0][0],
        "clip_genre_count": len(genres_sorted)
    }

    return res


if __name__ == "__main__":

    # argument parsing
    parser = argparse.ArgumentParser(description='clip genre estimation',
                                     epilog="stg7 2024",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", type=str, nargs="+", help="video to extract scores")
    parser.add_argument("--features_folder", type=str, default="features_clip_genre", help="only for calculate features, folder to store the features")

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
