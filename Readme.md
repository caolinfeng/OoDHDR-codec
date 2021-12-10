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

## Data Download

SDR training datasets can be downloaded from:
- [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K)
- [Flickr2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar)

HDR training datasets can be downloaded from:
- [HDRHEVEN](https://hdrihaven.com/hdris/)
- [pfstools(HDR Image Gallery)](http://pfstools.sourceforge.net/hdr_gallery.html)
- [HDRLabs](http://www labs.com/sibl/archive.htm)
- [Funt and Shi 2010](https://www2.cs.sfu.ca/~colour/data/funt_hdr/)

## Usage

### Training

A training script with a regularization & fusion loss is provided in
`examples/train_ood.py`. You can run the script for a training pipeline:

```bash
python examples/train_ood.py --lambda 12 --epochs 250 --cuda --save --gpu 0,1,2,3 --batch-size 32 --rw 0.00001 --pw 1 --sdr_w 0.95 
```
> **Note:** 'rw, pw, sdr_w' are the hyper-parameters of the constructed loss, to achevie the optimal performance of a certain network, it is recommended to use the grid search.
You can also modify other parameters to change the model and training strategy in the file or cmd.

### Evaluation

To evaluate a trained model on HDR and SDR dataset, evaluation scripts (`examples/test_hdr.py`, `examples/test_sdr.py`) are provided. Please modify the testing dataset path in the corresponding file, and specify the trained model path in cmd: 

```bash
python examples/test_hdr.py --cuda --pth /XXX.pth.tar
```
```bash
python examples/test_sdr.py --cuda --pth /XXX.pth.tar
```



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
 * This project is developed based on CompressAI library: https://github.com/InterDigitalInc/CompressAI
 * Kodak image dataset: http://r0k.us/graphics/kodak/

