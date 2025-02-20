# MCSANet

## Prerequisites

The following packages are required to run the scripts:
- [Python >= 3.8]
- [PyTorch >= 1.6]
- [Torchvision]
- [Pycocoevalcap]

## Datasets
We use two datasets (IU X-Ray and MIMIC-CXR) in our paper.

For `IU X-Ray`, you can download the dataset from [here](https://openi.nlm.nih.gov/faq).

For `MIMIC-CXR`, you can download the dataset from [here](https://physionet.org/content/mimic-cxr/2.0.0/).

Our experiments on IU X-Ray were done on a machine with 1x3090.

Our experiments on MIMIC-CXR were done on a machine with 1x3090.

## Pseudo Label Generation
You can generate the pesudo label for each dataset by leveraging the automatic labeler  [ChexBert](https://github.com/stanfordmlgroup/CheXbert).

## Acknowledgment
Our project references the codes in the following repos. Thanks for their works and sharing.
- [R2GenCMN](https://github.com/cuhksz-nlp/R2GenCMN)
