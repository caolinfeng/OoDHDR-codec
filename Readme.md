# OoDHDR-codec

This repository is for "OoDHDR-codec: Out-of-Distribution Generalization for HDR Image Compression"

(to appear in AAAI, 2022)

By Linfeng Cao, Aofan Jiang, Wei Li, Huaying Wu and Nanyang Ye


## Overview

<div align=center><img src="https://github.com/caolinfeng/OoDHDR-codec/blob/master/overview/framework.png" width="90%" height="90%"></div align=center>
<p align="center">Overview of the proposed OoDHDR-codec framework.</p>
<div align=center><img src="https://github.com/caolinfeng/OoDHDR-codec/blob/master/overview/dnn_backbone.png" width="90%" height="90%"></div align=center>
<p align="center">DNN backbone in our project.</p>

## Dependencies

- Python (3.8.3)
- PyTorch (>=1.6.0)
- torchvision
- matplotlib
- tensorboard

## Installation
**From source**:

```bash
git clone https://github.com/caolinfeng/OoDHDR-codec OodHDR_codec
cd OodHDR_codec
pip install -U pip && pip install -e .
```

## Data Download

SDR training datasets can be downloaded from:
- [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K)
- [Flickr2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar)

HDR training datasets (.hdr) can be downloaded from:
- [HDRHEVEN](https://hdrihaven.com/hdris/)
- [pfstools(HDR Image Gallery)](http://pfstools.sourceforge.net/hdr_gallery.html)
- [HDRLabs](http://www.hdrlabs.com/sibl/archive/)
- [Funt and Shi 2010](https://www2.cs.sfu.ca/~colour/data/funt_hdr/)

Evaluation datasets:
- [HDR]
- [SDR-Kodak](http://r0k.us/graphics/kodak/)

## Usage

### Training

A training script with a regularization & fusion loss is provided in
`examples/train_ood.py`. Please specify the corresponding HDR & SDR datasets in the file. The custom ImageFolder structure in our project only supports for the RGBE (.hdr) input currently.

You can run the script for a training pipeline:

```bash
python examples/train_ood.py --lambda 12 --epochs 250 --cuda --save --gpu 0,1,2,3 --batch-size 32 --rw 1e-5 --pw 1 --sdr_w 0.95 
```
> **Note:** 'rw, pw, sdr_w' are the hyper-parameters of the constructed loss. To achevie the optimal performance of a certain network, it is recommended to use the grid search.
You can also modify other parameters to change the model and training strategy in the file or cmd.

### Evaluation

To evaluate a trained model on HDR and SDR dataset, evaluation scripts (`examples/test_hdr.py`, `examples/test_sdr.py`) are provided. Please modify the testing dataset path in the corresponding file, and specify the trained model path in cmd: 

```bash
python examples/test_hdr.py --cuda --pth /XXX.pth.tar
```
```bash
python examples/test_sdr.py --cuda --pth /XXX.pth.tar
```

* The PyTorch pre-trained models can be downloaded [here (Google Drive)](https://drive.google.com/drive/folders/1FPUvTdN0JkoNJjm3FHDyrdPtLrddUxOm)

## Citation

If you use this project, please cite the relevant original publications for the
models and datasets, and cite this project as:

```

```

## Related links
 * This project is developed based on [CompressAI library](https://github.com/InterDigitalInc/CompressAI)

