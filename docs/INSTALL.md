# Installation

In this section we demonstrate how to prepare an environment with PyTorch.

## Requirements

- Linux (Windows is not officially supported)
- Python 3.6+
- PyTorch 1.8 or higher
- CUDA 10.1 or higher
- NCCL 2
- GCC 4.9 or higher
- [mmcv-full](https://github.com/open-mmlab/mmcv) 1.4.7 or higher (use `mmcv` for fast installation)

We have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04/18.04 and CentOS 7.2
- CUDA: 10.0/10.1/11.0/11.2
- NCCL: 2.1.15/2.2.13/2.3.7/2.4.2 (PyTorch-1.1 w/ NCCL-2.4.2 has a deadlock bug, see [here](https://github.com/open-mmlab/OpenSelfSup/issues/6))
- GCC(G++): 4.9/5.3/5.4/7.3/7.4/7.5

## Install OpenBioSeq

We recommend that users follow our best practices to install OpenBioSeq (similar to OpenMixup).

**Step 0.** Create a conda virtual environment and activate it.

```shell

conda create -n openbioseq python=3.8 -y

conda activate openbioseq

```

**Step 1.** Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g., on GPU platforms:

```shell

conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch

# assuming CUDA=10.1, "pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html"

```

**Step 2.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim). We recommend the users to install mmcv-full from the source using MIM (or it will install from the source with pip in step 4). You can also use `pip install mmcv` for fast installation.

```shell

pip install -U openmim

mim install mmcv-full

```

**Step 3.** Install other third-party libraries (not necessary). Please install opencv-contrib-python for SaliencyMix, `pip install opencv-contrib-python`. Please install [pyGCO](https://github.com/Borda/pyGCO) for PuzzleMix (used for cut_grid_graph, DON'T USE `pip install gco==1.0.1`).

**Step 4.** Install OpenBioSeq. To develop and run OpenBioSeq directly, install it from the source:

```shell

git clone https://github.com/Westlake-AI/OpenBioSeq.git

cd openbioseq

pip install -v -e .

# "-v" means verbose, and "-e" means installing in editable mode;

# or "python setup.py develop"

```

**Step 5.** Install Apex (optional), following the [official instructions](https://github.com/NVIDIA/apex), e.g.

```shell

git clone https://github.com/NVIDIA/apex

cd apex

pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

```

If some errors occur when you install Apex from the source, you can try `python setup.py install` for fast installation.

**Note:**

1. The git commit id will be written to the version number with step d, e.g. 0.1.0+2e7045c. The version will also be saved in trained models.
2. Following the above instructions, openmixup is installed on `dev` mode, any local modifications made to the code will take effect without the need to reinstall it (unless you submit some commits and want to update the version number).
3. If you would like to use `opencv-python-headless` instead of `opencv-python`,

you can install it before installing MMCV.

## Customized installation

### Benchmark

According to [MMSelfSup](https://github.com/open-mmlab/mmselfsup), if you need to evaluate your pre-training model with some downstream tasks such as detection or segmentation, please also install [Detectron2](https://github.com/facebookresearch/detectron2), [MMDetection](https://github.com/open-mmlab/mmdetection) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).

If you don't run MMDetection and MMSegmentation benchmark, it is unnecessary to install them.

You can simply install MMDetection and MMSegmentation with the following command:

```shell

pip install mmdet mmsegmentation

```

For more details, you can check the installation page of [MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md).

### Install MMCV without MIM

MMCV contains C++ and CUDA extensions, thus depending on PyTorch in a complex way. MIM solves such dependencies automatically and makes the installation easier. However, it is not a must. To install MMCV with pip instead of MIM, please follow [MMCV installation guides](https://mmcv.readthedocs.io/en/latest/get_started/installation.html). This requires manually specifying a find-url based on PyTorch version and its CUDA version.

For example, the following command install mmcv-full built for PyTorch 1.10.x and CUDA 11.3.

```shell

pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html

```

### Another option: Docker Image

We provide a [Dockerfile](/docker/Dockerfile) to build an image.

```shell

# build an image with PyTorch 1.10.0, CUDA 11.3, CUDNN 8.

docker build -f ./docker/Dockerfile --rm -t openbioseq:torch1.10.0-cuda11.3-cudnn8 .

```

**Note:** Make sure you've installed the [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

## Prepare datasets

It is recommended to symlink your dataset root (assuming `$YOUR_DATA_ROOT`) to `$OPENBIOSEQ/data`.

If your folder structure is different, you may need to change the corresponding paths in config files.

### Prepare Classification Datasets

We support following datasets: CIFAR-10/100, [Tiny-ImageNet](https://www.kaggle.com/c/tiny-imagenet), [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/), [Place205](http://places.csail.mit.edu/downloadData.html), [iNaturalist2017/2018](https://github.com/visipedia/inat_comp), [CUB200](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [FGVC-Aircrafts](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/), [StandordCars](http://ai.stanford.edu/~jkrause/cars/car_dataset.html). Taking ImageNet for example, you need to 1) download ImageNet; 2) create the following list files or download [meta](https://github.com/Westlake-AI/openmixup/releases/download/dataset/meta.zip) under $DATA/meta/: `train.txt` and `val.txt` contains an image file name in each line, `train_labeled.txt` and `val_labeled.txt` contains `filename label\n` in each line; `train_labeled_*percent.txt` are the down-sampled lists for semi-supervised evaluation. 3) create a symlink under `$OPENMIXUP/data/`. See OpenMixup for more details for image datasets.

## A from-scratch setup script

Here is a full script for setting up openmixup with conda and link the dataset path. The script does not download full datasets, you have to prepare them on your own.

```shell
conda create -n openbioseq python=3.8 -y
conda activate openbioseq
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 # as an example

git clone https://github.com/Westlake-AI/OpenBioSeq.git
cd openbioseq

pip install openmim
mim install mmcv-full
python setup.py develop

# download 'meta' and move to `data/meta` for image datasets
wget https://github.com/Westlake-AI/openmixup/releases/download/dataset/meta.zip
unzip -d data/meta meta.zip

# download full classification datasets
ln -s $CIFAR10_ROOT data/cifar10
ln -s $CIFAR100_ROOT data/cifar100
ln -s $IMAGENET_ROOT data/ImageNet
ln -s $TINY_ROOT data/TinyImagenet

# download VOC datasets
bash tools/prepare_data/prepare_voc07_cls.sh $YOUR_DATA_ROOT

```

## Common Issues

1. The installation error might occur with high version of PyTorch and MMCV-full, `error: urllib3 2.0.3 is installed but urllib3<2.0 is required by {'google-auth'}`. You can tackle the error by `pip install urllib3==2.0.3 google-auth` and `python setup.py develop`.
2. The training hangs / deadlocks in some intermediate iteration. See this [issue](https://github.com/open-mmlab/OpenSelfSup/issues/6). We haven't found this issue when using higher versions of PyTorch.
