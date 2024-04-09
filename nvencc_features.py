#!/usr/bin/env python3
import os
import re
import argparse
import json

from utils import shell_call


def parse_nvenc(res):
    infos = {
        "iframes_ratio": 0,
        "pframes_ratio": 0,
        "bframes_ratio": 0
    }
    get_avg_qp = lambda x: float(re.search("avgQP(.*),.*", x).group(1).strip())
    get_num_frames = lambda x: int(re.search("type .(.*?),.*", x).group(1).strip())

    for l in res.split("\n"):
        if "frame type I" in l and "avgQP" in l:
            infos["avg_iframe_qp"] = get_avg_qp(l)
            infos["iframes_ratio"] = get_num_frames(l)
        if "frame type P" in l:
            infos["avg_pframe_qp"] = get_avg_qp(l)
            infos["pframes_ratio"] = get_num_frames(l)
        if "frame type B" in l:
            infos["avg_bframe_qp"] = get_avg_qp(l)
            infos["bframes_ratio"] = get_num_frames(l)
        if "ssim/psnr/vmaf: SSIM YUV:" in l:
            infos["ssim"] = float(re.search("All:(.*?)\(.*", l).group(1).strip())
        if "ssim/psnr/vmaf: PSNR YUV:" in l:
            infos["psnr"] = float(re.search("Avg:(.*?)\,.*", l).group(1).strip())
        if "Input Info" in l:
            infos["height"] = float(re.search(",(.*)x(.*),.*", l).group(2).strip())
            infos["width"] = float(re.search(",(.*)x(.*),.*", l).group(1).strip())
            infos["aspect"] =  infos["width"] / infos["height"]
            infos["fps"] = round(eval(re.search(".*,(.*) fps.*", l).group(1).strip()), 2)
        if "encoded" in l:
            infos["bitrate"] = float(re.search(".*fps, (.*) kbps,.*", l).group(1).strip())
    num_frames = infos["iframes_ratio"] + infos["pframes_ratio"] + infos["bframes_ratio"]
    infos["iframes_ratio"] /= num_frames
    infos["iframes_ratio"] = round(infos["iframes_ratio"], 4)
    infos["pframes_ratio"] /= num_frames
    infos["pframes_ratio"] = round(infos["pframes_ratio"], 4)
    infos["bframes_ratio"] /= num_frames
    infos["bframes_ratio"] = round(infos["bframes_ratio"], 4)

    return infos


def extract_features(video):
    video_path_tmp = os.path.basename(video) + "_tmp.mp4"
    cmd = f"""ffmpeg -loglevel quiet -i {video} -strict -1 -f yuv4mpegpipe -  |  (./nvencc --y4m -i - --ssim --psnr --codec h265 -o {video_path_tmp} 2>&1)"""
    #print(cmd)#--frames <INT>

    res = shell_call(cmd)
    encoding_features = parse_nvenc(res)

    os.remove(video_path_tmp)
    return encoding_features



if __name__ == "__main__":

    # argument parsing
    parser = argparse.ArgumentParser(description='nvenc features estimation',
                                     epilog="stg7 2024",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", type=str, nargs="+", help="video to extract scores")
    parser.add_argument("--features_folder", type=str, default="features_nvenc", help="only for calculate features, folder to store the features")

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


