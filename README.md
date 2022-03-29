# MDX-Net Music Source Separation

This repository contains the source code and pretrained models for MDX-Net's submission to the [Sony Demixing Challenge](https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021), which ranked second place.

This branch is undergoing refractorization, the code may break for you. Please see [Reproducible Submission](#for-musicians-reproducible-submission) for branches that contain working source code and pretrained models.

## For machine learning scientists

Please see [MDX-Net](https://github.com/kuielab/mdx-net) for the code to train your own models.


## For musicians (Reproducible Submission)

Please see the following branches for instructions to install and use the pretrained models.
- MDX Track A submission: [leaderboard_A](https://github.com/kuielab/mdx-net-submission/tree/leaderboard_A)
	- Uses only [MUSDB18-HQ dataset](https://zenodo.org/record/3338373), lower quality stems extracted
- MDX Track B submission: [leaderboard_B](https://github.com/kuielab/mdx-net-submission/tree/leaderboard_B)
	- Uses extra datasets, higher quality stems extracted


## Installation

```
conda env create -f environment.yml -n mdx-submit
pip install -r requirements.txt
```