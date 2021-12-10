# OoDHDR-codec

This repository is for "OoDHDR-codec: Out-of-Distribution Generalization for HDR Image Compression"

(to appear in AAAI, 2022)

By Linfeng Cao, Aofan Jiang, Wei Li, Huaying Wu and Nanyang Ye


## Overview

<div align=center><img src="https://github.com/caolinfeng/OoDHDR-codec/blob/master/overview/framework.png" width="90%" height="90%"></div align=center>
<p align="center">Overview of the proposed OoDHDR-codec framework.</p>
<div align=center><img src="https://github.com/caolinfeng/OoDHDR-codec/blob/master/overview/dnn_backbone.png" width="90%" height="90%"></div align=center>
<p align="center">DNN backbone used in our project.</p>


## Installation
**From source**:

```bash
https://github.com/caolinfeng/OoDHDR-codec.git
cd OoDHDR-codec
pip install -U pip && pip install -e .
```

## Usage

### Training

A training script with a regularization & fusion loss is provided in
`examples/train_ood.py`. You can run the script for a training pipeline:

```bash
python examples/train_ood.py --lambda 12 --epochs 250 --cuda --save --gpu 0,1,2,3 --batch-size 32 --rw 0.00001 --pw 1 --sdr_w 0.95 
```
> **Note:** 'rw, pw, sdr_w' are the hyper-parameters of the constructed loss, to achevie the optimal performance of a certain network, it is recommended to use the grid search.
You can also modify other parameters to change the model and training strategy in the way.

### Evaluation

To evaluate a trained model on HDR and SDR dataset, evaluation scripts (`examples/test_hdr.py`, `examples/test_sdr.py`) are provided. Please modify the testing dataset path in the corresponding file, and specify the trained model path in cmd: 

```bash
python examples/test_hdr.py --cuda --pth /XXX.pth.tar
```
```bash
python examples/test_sdr.py --cuda --pth /XXX.pth.tar
```


## License

CompressAI is licensed under the Apache License, Version 2.0

## Contributing

We welcome feedback and contributions. Please open a GitHub issue to report
bugs, request enhancements or if you have any questions.

Before contributing, please read the CONTRIBUTING.md file.

## Authors

* Jean Bégaint, Fabien Racapé, Simon Feltman and Akshay Pushparaja, InterDigital AI Lab.

## Citation

If you use this project, please cite the relevant original publications for the
models and datasets, and cite this project as:

```
@article{begaint2020compressai,
	title={CompressAI: a PyTorch library and evaluation platform for end-to-end compression research},
	author={B{\'e}gaint, Jean and Racap{\'e}, Fabien and Feltman, Simon and Pushparaja, Akshay},
	year={2020},
	journal={arXiv preprint arXiv:2011.03029},
}
```

## Related links
 * Tensorflow compression library by _Ballé et al._: https://github.com/tensorflow/compression
 * Range Asymmetric Numeral System code from _Fabian 'ryg' Giesen_: https://github.com/rygorous/ryg_rans
 * BPG image format by _Fabrice Bellard_: https://bellard.org/bpg
 * HEVC HM reference software: https://hevc.hhi.fraunhofer.de
 * VVC VTM reference software: https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM
 * AOM AV1 reference software: https://aomedia.googlesource.com/aom
 * Z. Cheng et al. 2020: https://github.com/ZhengxueCheng/Learned-Image-Compression-with-GMM-and-Attention
 * Kodak image dataset: http://r0k.us/graphics/kodak/

