## MDX-Net Track A Submission

This branch contains the source code and the pretrained model that is submitted to the [Sony MDX Challenge](https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021) Track A.

## Precautions

<div style="size: 2em; font-weight: bold;">
The repository supports Linux, MacOS (Intel and M1) and Windows. <br />
The separated stems each have a different <a href="https://ws-choi.github.io/personal/presentations/slide/2021-08-21-aicrowd#/2/1">frequency cutoff</a> applied. This is inherent to the approach of the MDX-Net code, which means that you would not obtain lossless stem tracks as compared to the original.
</div>

## Installation

Set up [conda](https://docs.conda.io/en/latest/miniconda.html) first. After it's installed, clone this branch:

```bash
git clone -b leaderboard_A https://github.com/kuielab/mdx-net-submission.git
```

### Linux

In the cloned repository directory, do

```bash
conda env create -f environment.yml -n mdx-submit
conda activate mdx-submit
pip install -r requirements.txt
python download_demucs.py
wget https://zenodo.org/record/5717356/files/onnx_A.zip
unzip onnx_A
wget https://zenodo.org/record/5717356/files/mixer.ckpt
mv mixer.ckpt model
```

### MacOS

For Intel macs, the procedure is the same as [that on Linux](#linux).

For M1 macs, use `environment-m1.yml` in the first command instead of `environment.yml`.

If it errors out with not finding `libsndfile`, install it using your preferred package manager and put `libsndfile.dylib` into the directory it tells you.

### Windows

In the cloned repository directory, using the conda powershell prompt:

```bash
conda env create -f environment.yml -n mdx-submit
conda activate mdx-submit
pip install -r requirements.txt
python download_demucs.py
Invoke-WebRequest -Uri https://zenodo.org/record/5717356/files/onnx_A.zip -OutFile onnx_A.zip
Expand-Archive onnx_A.zip -DestinationPath .
Invoke-WebRequest -Uri https://zenodo.org/record/5717356/files/mixer.ckpt -OutFile ./model/mixer.ckpt
```

## Custom models

For custom models (such as the [higher quality vocal model trained by UVR team](https://github.com/Anjok07/ultimatevocalremovergui/releases/tag/MDX-Net-B)), please replace the relevant models in `./onnx_A/`.

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
