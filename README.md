## MDX-Net Track B Submission

This branch contains the source code and the pretrained model that is submitted to the [Sony MDX Challenge](https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021) Track B.

## Precautions

<div style="size: 2em; font-weight: bold;">
The repository supports Linux and <a href="https://github.com/kuielab/mdx-net-submission/issues/1">does not support Windows</a> (MacOS untested). <br />
The separated stems each have a different <a href="https://ws-choi.github.io/personal/presentations/slide/2021-08-21-aicrowd#/2/1">frequency cutoff</a> applied. This is inherent to the approach of the MDX-Net code, which means that you would not obtain lossless stem tracks as compared to the original.
</div>

## Installation

Set up [conda](https://docs.conda.io/en/latest/miniconda.html) first. After it's installed, clone this branch:

```bash
git clone -b leaderboard_B https://github.com/kuielab/mdx-net-submission.git
```

### Linux

In the cloned repository directory, do

```bash
conda env create -f environment.yml -n mdx-submit
conda activate mdx-submit
pip install -r requirements.txt
python download_demucs.py
wget https://zenodo.org/record/6513062/files/onnx_B.zip
unzip onnx_B
```

### Windows

In the cloned repository directory, using the conda powershell prompt:

```bash
conda env create -f environment.yml -n mdx-submit
conda activate mdx-submit
pip install -r requirements.txt
python download_demucs.py
Invoke-WebRequest -Uri https://zenodo.org/record/6513062/files/onnx_B.zip -OutFile onnx_B.zip
Expand-Archive onnx_B.zip -DestinationPath .
```

## Usage

After successful installation, you can put the songs you wish to separate as `./data/test/SONGNAME/mixture.wav`, and run either `run.sh` or

```bash
python predict_blend.py
```

After the separation completes, the results will be saved in `./data/results/kuielab_mdxnet_A/SONGNAME/`.

Also, every time when you open a new terminal / conda prompt, conda will default to environment `base`.
Just do 

```bash
conda activate mdx-submit
```

to go back into the environment you have installed MDX's dependencies in.
