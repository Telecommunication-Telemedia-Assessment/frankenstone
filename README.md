# frankenstone

This is the final model for the "UGC Video Quality Assessment Challenge".

## design approach
The frankenstone model uses several other models/features as a baseline.
E.g. NVENCC will be used to extract meta-data and 'encoding' properties (such as bitrate for a specific encoding setting).
Furthermore, the [dover model](https://github.com/VQAssessment/DOVER) will be used (the overall score, and two atomic values).
In addition signal based features (e.g. SI, TI, ..) for a subset of the frames are extracted and furthermore for the same subset also [VILA model](https://github.com/google-research/google-research/tree/master/vila) predictions (image appeal) are performed.
The subset is for each second the first frame of the video, and then another reduction of the resulting frames considering that the beginning of the video is less important.
All features are extracted in separated threads to make the model faster.
Afterwards the features are combined using a Random Forest Regression model.


## requirements

* ubuntu >=22.04
* nvencc should run (binary is included)
* ffmpeg install via conda (see `conda base setup`) or globally
* python 3.11 in a conda environment with cuda (see `conda base setup`) 
* python dependencies: `python3 -m pip install -r requirements.txt`

* run `./prepare.sh` to extract the pretrained models (stored in `pretrained_weights.tar.lzma`, extraction needs lzma and tar installed)

* **Important** to run the model, you need a GPU, e.g. Nvidia 3090 with at least 12 GB of GPU memory (for 4K videos), tested with Nvidia 3090 Ti 24 GB GPU Ram.

### conda base setup

```bash

conda create -n ENV python=3.11
conda activate ENV
conda install cudnn cudatoolkit ffmpeg

cd $CONDA_PREFIX
mkdir -p ./etc/conda/activate.d
touch ./etc/conda/activate.d/env_vars.sh
```

edit ./etc/conda/activate.d/env_vars.sh as follows:
```bash
#!/bin/sh
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
```

deactivate and reactivate conda env
```bash
conda deactivate
conda activate ENV
```

install tensorflow with pip (see https://www.tensorflow.org/install/pip)

```bash
python3 -m pip install tensorflow[and-cuda]
```

check if gpu/gpus are detected:
```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
```

this command should show the gpu/gpus, and should in the best case not complain, that "something is missing"


## usage

A call of `./frankenstone.py --help` should result in: (important run it within your conda environment)

```
usage: frankenstone.py [-h] [--features_only] [--features_folder FEATURES_FOLDER] video [video ...]

predict ugc video quality (using GPU) in a frankenstone approach

positional arguments:
  video                 video to process

options:
  -h, --help            show this help message and exit
  --features_only, -fo  only calculate features, no model prediction (default: False)
  --features_folder FEATURES_FOLDER
                        only for calculate features, folder to store the features (default: features)

stg7 2024
```

## model training
To train the model you need the YTUGC dataset (e.g. in ./ugc) and the `original_videos` sub-folder (thus uncompressed files), which will be in the following referred to as `<UGCFOLDER>`.

To extract the features run:
```bash
find <UGCFOLDER> -name "*.mkv" | xargs -i ./frankenstone.py -fo {}
```
(you can modify `xargs -i` to `xargs -i -P <X>` with `<X>` number of parallel calculations, on the test system 2-3 was possible)
Check afterwards if in `features` for all videos features were extracted.
Change to the `model` directory and run `./model.py`, this will train the final model using the training data and also performs the prediction on the test data (stored as `ais24_pred.csv`).

## run time
The run time of the model has been evaluated exemplary with the `Sports_2160P-210c.mkv` (~30 fps, UHD-1, 20s duration) video.
The model should be faster than 20s, however, this depends on the used SSD (to load the models and video to GPU memory) and used memory, cpu.

Example run:
```
time ./frankenstone.py Sports_2160P-210c.mkv 
``` 
resulted in 
```
{"quality": 4.326934337615967, "video": "Sports_2160P-210c.mkv"}
calculation done for main: 15.439289331436157 s; 

real    0m19,950s
user    0m40,747s
sys     0m22,085s

```


Note: the YTUGC dataset has a variety of videos included, thus there are videos with a higher resolution than UHD-1 (some of the VR videos), with 60 fps, with lower fps.


## origin of the name
frankenstone is a bad translation of frankenstein, which is a reference to the Frankenstein novel by Mary Shelley.
There Frankenstein's monster is somehow "put together" by different pieces, thus the proposed model is similar to this "hacking together" approach.