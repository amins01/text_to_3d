# Text to 3D (under 2 minutes)
This repository provides a fast and simple text to 3D generation framework using Gaussian Splatting, based on the [DreamGaussian paper](https://arxiv.org/pdf/2309.16653).

https://github.com/user-attachments/assets/616781a3-a457-4922-a92e-467236ad867c

## Qualitative Comparison
The following section presents a qualitative comparison of our results with the ones of DreamFusion, Shape-E, DreamGaussian:

![qualitative_comparison](https://github.com/user-attachments/assets/baf441fd-d79c-463d-8d7c-1ede942484a9)

## Ablation Study
The following ablation study shows the impact of different settings on generating a 3D model from the prompt "a photo of a hamburger":

![ablation_study](https://github.com/user-attachments/assets/2b6a2c16-9f3e-4d5e-951d-7b17c13e7247)

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
