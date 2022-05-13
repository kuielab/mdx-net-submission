## MDX-Net Track B Submission

This branch contains the source code and the pretrained model that is submitted to the [Sony MDX Challenge](https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021) Track B.

## Precautions

<div style="size: 2em; font-weight: bold;">
The repository supports Linux and MacOS(also on m1 Mac), but <a href="https://github.com/kuielab/mdx-net-submission/issues/1">does not support Windows</a> . <br />
The separated stems each have a different <a href="https://ws-choi.github.io/personal/presentations/slide/2021-08-21-aicrowd#/2/1">frequency cutoff</a> applied. This is inherent to the approach of the MDX-Net code, which means that you would not obtain lossless stem tracks as compared to the original.
</div>

## Installation

Setup [git-lfs](https://git-lfs.github.com/) first. You will need it to download the models inside this repository.
You'd also need [conda](https://docs.conda.io/en/latest/miniconda.html).

After all those are installed, clone this branch:

```bash
git clone -b leaderboard_B https://github.com/kuielab/mdx-net-submission.git
```

In the cloned repository directory, do

```bash
conda env create -f environment.yml -n mdx-submit
conda activate mdx-submit
pip install -r requirements.txt
python download_demucs.py
```

Specially, when using a m1 mac, change the first step as below shows, as we will use onnxruntime-silicon instead of onnxruntime. If encounter the problem of libsndfile, just install libsndfile(use package managers like homebrew is preferred) and put libsndfile.dylib into the dirctory it tells you.

```bash
conda env create -f environment-m1.yml -n mdx-submit
```


Every time when you open a new terminal, conda will default to environment `base`.
Just do 

```bash
conda activate mdx-submit
```

to go back into the environment you have installed MDX's dependencies in.

## Custom models

For custom models (such as the [higher quality vocal model trained by UVR team](https://github.com/Anjok07/ultimatevocalremovergui/releases/tag/MDX-Net-B)), please replace the relevant models in `./onnx/`.

## Usage

After successful installation, you can put the songs you wish to separate as `./data/test/SONGNAME/mixture.wav`, and run either `run.sh` or

```bash
python predict_blend.py
```

After the separation completes, the results will be saved in `./data/results/baseline/SONGNAME/`.
