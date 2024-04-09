#!/usr/bin/env python3
import subprocess
import time
import sys

import numpy as np
import decord
from decord import VideoReader


def video_frames(video_path, bridge="tensorflow"):
    decord.bridge.set_bridge(bridge)
    vr = VideoReader(video_path)
    fps = round(vr.get_avg_fps())
    frames = []
    for x in range(0, len(vr), fps):
        frames.append(vr[x])
    return frames


def sample_non_uniform(a, k=3):
    # divides an array into k groups, samples first k elements, then after len(a)/k elements k-1 elements and so on...
    i = 0
    remaining = len(a)
    idx = []
    while i < len(a):
        idx.append(i)
        if i > remaining // k and k > 1:
            k -= 1
        i += k
        remaining -= k
    print("selected frame idx", idx)
    return [a[i] for i in idx]


def merge_dicts(dicts):
    rr = {}
    for r in dicts:
        rr = dict(rr, **r)
    return rr


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


def weighted_mean(res):
    return np.sum([ x * (i+1) for i, x in enumerate(res)])


def shell_call(call):
    """
    Run a program via system call and return stdout + stderr.
    @param call programm and command line parameter list, e.g shell_call("ls /")
    @return stdout and stderr of programm call
    """
    try:
        output = subprocess.check_output(
            call, universal_newlines=True, shell=True
        )
    except Exception as e:
        output = str(e.output)
    return output


def timeit(func):
    """
    get the required time for a function call as information on stdout, via lInfo

    example usage:

    .. code-block:: python

        @timeit
        def myfunction():
            return 42

    or

    .. code-block:: python

        def anotherfucntion():
            return 23

        timeit(anotherfucntion())

    """

    def function_wrapper(*args, **kwargs):
        start_time = time.time()
        print("start {}".format(func.__name__), file=sys. stderr)
        res = func(*args, **kwargs)
        overall_time = time.time() - start_time
        print(
            "calculation done for {}: {} s; ".format(
                func.__name__,
                overall_time,
                #str(datetime.timedelta(seconds=overall_time)),
            ), file=sys.stderr
        )
        return res

    return function_wrapper