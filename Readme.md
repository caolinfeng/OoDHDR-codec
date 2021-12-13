# OoDHDR-codec

This repository is for "OoDHDR-Codec: Out-of-Distribution Generalization for HDR Image Compression"

(to appear in AAAI, 2022)

By Linfeng Cao, Aofan Jiang, Wei Li, Huaying Wu and Nanyang Ye

- Paper full supplementary is available [here (Google Drive)](https://drive.google.com/file/d/1HYOSB4owuOZWVKBzTkJRylLeumL1xN4Y/view?usp=sharing)
## Overview

<div align=center><img src="https://github.com/caolinfeng/OoDHDR-codec/blob/master/overview/framework.png" width="90%" height="90%"></div align=center>
<p align="center">Overview of the proposed OoDHDR-codec framework.</p>
<div align=center><img src="https://github.com/caolinfeng/OoDHDR-codec/blob/master/overview/dnn_backbone.png" width="90%" height="90%"></div align=center>
<p align="center">DNN backbone in our project.</p>

## Dependencies

- Python (3.8.3)
- PyTorch (>=1.6.0)

 (see `setup.py` for the full list)


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
- [HDR](https://drive.google.com/drive/folders/1U_GN4UARkLFve3UjvRhNKs56z6dcH-WI?usp=sharing)
- [SDR-Kodak](http://r0k.us/graphics/kodak/)

## Usage

### Training

A training script with a regularization & fusion loss is provided in
`examples/train_ood.py`. Please specify the corresponding HDR & SDR datasets in the file. The custom ImageFolder structure in our project only supports for the RGBE (.hdr) input currently.

You can run the script for a training pipeline:

```bash
python examples/train_ood.py --lambda 12 --epochs 250 --cuda --save --gpu 0,1,2,3 --batch-size 32 --rw 1e-5 --pw 2 --sdr_w 0.95 
```
> **Note:** 'rw, pw, sdr_w' are the hyper-parameters of the constructed loss. To achevie the optimal performance of a certain network, it is recommended to use the grid search.
You can also modify other parameters to change the model and training strategy in the file or cmd.

### Evaluation

To evaluate a trained model on HDR and SDR dataset, evaluation scripts (`examples/test_hdr.py`, `examples/test_sdr.py`) are provided. Please modify the testing dataset path in the corresponding file, and specify the trained model path in cmd: 

```bash
python examples/test_hdr.py --pth /XXX.pth.tar
```
```bash
python examples/test_sdr.py --pth /XXX.pth.tar
```

* The PyTorch pre-trained models can be downloaded [here (Google Drive)](https://drive.google.com/drive/folders/1FPUvTdN0JkoNJjm3FHDyrdPtLrddUxOm)

### Quality Assessement

To assess the compression performance on HDR images, the evaluation metrics of puPSNR, puSSIM and HDR-VDP (3.0.6) are used, the source codes (Matlab version) can be [downloaded here](http://resources.mpi-inf.mpg.de/hdr/vdp/). 

## Citation

If you find this code useful, please cite our paper::

```
@inproceedings{Cao2020OodHDR,
  title     = {OoDHDR-Codec: Out-of-Distribution Generalization for HDR Image Compression},
  author    = {Linfeng Cao, Aofan Jiang, Wei Li, Huaying Wu and Nanyang Ye},
  booktitle = {Proceedings ofthe Fourth National Conference on Artificial Intelligence},
  year      = {2022}
}
```

## Related link
 * This project is developed based on [CompressAI library](https://github.com/InterDigitalInc/CompressAI)

