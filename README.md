# frankenstone toolbox

This is the frankenstone toolbox to evaluate and unify models and features for user-generated video quality.


## requirements

* ubuntu >=22.04
* nvencc should run (binary is included)
* ffmpeg install via conda (see `conda base setup`) or globally
* python 3.11 in a conda environment with cuda (see `conda base setup`) 
* python dependencies: `python3 -m pip install -r requirements.txt`

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
TODO
```




## origin of the name
frankenstone is a bad translation of frankenstein, which is a reference to the Frankenstein novel by Mary Shelley.
There Frankenstein's monster is somehow "put together" by different pieces, thus the proposed toolbox is similar to this "hacking together" approach.


## acknowledgments
If you use this software in your research, please include a link to the repository and reference the following paper.

```bibtex
@article{goering2024frankenstone,
  title={The Frankenstone toolbox for video quality analysis
of user-generated content.},
  author={Steve G\"oring, Alexander Raake},
}
```