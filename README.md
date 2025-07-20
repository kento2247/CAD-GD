# CAD_GD-CVPR25
(CVPR25) This repository is the official implementation of our AAAI 2024 Paper [Exploring Contextual Attribute Density in Referring Expression Counting](https://arxiv.org/abs/2503.12460)

## Installation
Our code has been tested on Python 3.10 and PyTorch 2.4.0. 

1. Install [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO).
2. Install requirements.txt.

## Data Preparation
We train and evaluate our methods on REC8K and FSC-147 dataset. Please follow the REC8K and FSC-147 official repository to download and unzip the dataset.

* [REC-8k](https://github.com/sydai/referring-expression-counting)
* [FSC147](https://github.com/cvlab-stonybrook/LearningToCountEverything)

## Inference
You can run following command to conduct the inference on the REC-8k and FSC-147 dataset.

```
python test_rec.py
python test_fsc.py
```

## Training
We use the pretrained model from GroundingDINO, please download the pretrained weight from [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO). Then you can run the following command to conduct the traininng on the REC-8k or FSC-147 dataset.

```
python train_rec8k.py
python train_fsc.py
```

## Citation
If you find this work or code useful for your research, please cite:

```
@inproceedings{wang2025exploring,
  title={Exploring Contextual Attribute Density in Referring Expression Counting},
  author={Wang, Zhicheng and Pan, Zhiyu and Peng, Zhan and Cheng, Jian and Xiao, Liwen and Jiang, Wei and Cao, Zhiguo},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={19587--19596},
  year={2025}
}
```
