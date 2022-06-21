# OpenBioSeq

**News**

* OpenBioSeq v0.1.1 is released, which supports classification and regression tasks on bio-sequence datasets. It inherited most features in [OpenMixup](https://github.com/Westlake-AI/openmixup).
* OpenBioSeq v0.1.0 is initialized.

## Introduction

The main branch works with **PyTorch 1.8** or higher.

`OpenBioSeq` is an open-source supervised and self-supervised bio-sequence representation learning toolbox based on PyTorch. `OpenBioSeq` supports popular backbones, pre-training methods, and various features.

### What does this repo do?

Learning useful bio-sequence representation efficiently facilitates various downstream tasks in biological and chemical fields. This repo focuses on supervised and self-supervised bio-sequence representation learning and is named `OpenBioSeq`.

### Major features

This repo will be continued to update in the next two months! Please watch us for latest update!

## Change Log

Please refer to [CHANGELOG.md](docs/CHANGELOG.md) for details and release history.

[2020-06-09] `OpenBioSeq` v0.1.1 is released.

[2020-05-24] `OpenBioSeq` v0.1.0 is initialized.

## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for installation and dataset preparation.

## Get Started

Please see [Getting Started](docs/GETTING_STARTED.md) for the basic usage of OpenBioSeq (based on OpenMixup and MMSelfSup).
Then, see [tutorials](docs/tutorials) for more tech details (based on MMClassification), which is similar to most open-source projects in MMLab.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement

- `OpenBioSeq` is an open-source project for supervised and self-supervised methods on bio-sequence datasets created by researchers in CAIRI AI LAB. We encourage researchers interested in bio-sequence research and applications to contribute to `OpenBioSeq`!
- This repo is mainly based on [OpenMixup](https://github.com/Westlake-AI/openmixup), and borrows the architecture design and part of the code from [MMSelfSup](https://github.com/open-mmlab/mmselfsup) and [MMClassification](https://github.com/open-mmlab/mmclassification).

## Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@misc{2022openbioseq,
    title={{OpenBioSeq}: Open Toolbox and Benchmark for Bio-sequence Representation Learning},
    author={Li, Siyuan and Liu, Zichen and Wu, Di and Stan Z. Li},
    howpublished = {\url{https://github.com/Westlake-AI/openbioseq}},
    year={2022}
}
```

## Contributors

For now, the direct contributors include: Siyuan Li ([@Lupin1998](https://github.com/Lupin1998)) and Zicheng Liu ([@pone7](https://github.com/pone7)). We thanks contributors for OpenMixup, MMSelfSup, and MMClassification.

## Contact

This repo is currently maintained by Siyuan Li (lisiyuan@westlake.edu.cn) and Zicheng Liu (liuzicheng@westlake.edu.cn).
