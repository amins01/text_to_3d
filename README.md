# Text to 3D (under 2 minutes)
This repository provides a fast and simple text to 3D generation framework using Gaussian Splatting, based on the [DreamGaussian paper](https://arxiv.org/pdf/2309.16653).

## Qualitative Comparison
The following compares our results with the ones of DreamFusion, Shape-E, DreamGaussian:

## Ablation Study
The following ablation study shows the impact of different settings on generating a 3D model from the prompt "a photo of a hamburger":

## Installation
```
pip install -r requirements.txt

git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

pip install ./simple-knn
```

## Usage
1. Run the `main.py` script
2. Enter your prompt
3. Enjoy a real-time animation showing the 3D Gaussian splats adjusting to form a 3D model that matches your prompt

Example
```sh
$ python main.py
Enter your prompt: an icecream cone
```

## Acknowledgements
This project was made possible thanks to the contributions and inspiration from the following projects:
- [dreamgaussian](https://github.com/dreamgaussian/dreamgaussian)
- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)