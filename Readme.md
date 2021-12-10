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

### Examples

Script and notebook examples can be found in the `examples/` directory.

To encode/decode images with the provided pre-trained models, run the
`codec.py` example:

```bash
python3 examples/codec.py --help
```

An examplary training script with a rate-distortion loss is provided in
`examples/train.py`. You can replace the model used in the training script
with your own model implemented within CompressAI, and then run the script for a
simple training pipeline:

```bash
python3 examples/train.py -d /path/to/my/image/dataset/ --epochs 300 -lr 1e-4 --batch-size 16 --cuda --save
```
> **Note:** the training example uses a custom [ImageFolder](https://interdigitalinc.github.io/CompressAI/datasets.html#imagefolder) structure.

A jupyter notebook illustrating the usage of a pre-trained model for learned image
compression is also provided in the `examples` directory:

```bash
pip install -U ipython jupyter ipywidgets matplotlib
jupyter notebook examples/
```

### Evaluation

To evaluate a trained model on your own dataset, CompressAI provides an
evaluation script:

```bash
python3 -m compressai.utils.eval_model checkpoint /path/to/images/folder/ -a $ARCH -p $MODEL_CHECKPOINT...
```

To evaluate traditional image/video codecs:

```bash
python3 -m compressai.utils.bench --help
python3 -m compressai.utils.bench bpg --help
python3 -m compressai.utils.bench vtm --help
```

## Tests

Run tests with `pytest`:

```bash
pytest -sx --cov=compressai --cov-append --cov-report term-missing tests
```

Slow tests can be skipped with the `-m "not slow"` option.


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

