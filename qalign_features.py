#!/usr/bin/env python3

import os
import sys
import json

os.environ["TORCH_HOME"] = os.path.join(os.path.dirname(__file__), "pretrained_weights")

from q_align import QAlignVideoScorer
from q_align.evaluate.scorer import load_video


def extract_features(video_path, frame_sampling=True):
    scorer = QAlignVideoScorer() #(device="cpu")
    video_list = [load_video(sys.argv[1])]
    res = {
        "video": sys.argv[1],
        "score": float(scorer(video_list).tolist()[0])
    }
    return {"q_align_score": res["score"]}
