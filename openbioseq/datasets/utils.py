from collections.abc import Sequence

from PIL import Image
import mmcv
import numpy as np
import pickle
import torch
import torchvision


def to_numpy(data):
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, Image.Image):
        data = np.array(data, dtype=np.uint8)
        if data.ndim < 3:
            data = np.expand_dims(data, axis=-1)
        data = np.rollaxis(data, 2)  # HWC to CHW
        return data
    elif isinstance(data, torch.Tensor):
        return data.type(torch.float32).numpy()
    else:
        raise TypeError(
            f'Type {type(data)} cannot be converted to numpy.')


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).type(torch.float32)
    elif isinstance(data, Image.Image):
        return torchvision.transforms.functional.to_tensor(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(
            f'Type {type(data)} cannot be converted to tensor.'
            'Supported types are: `numpy.ndarray`, `torch.Tensor`, '
            '`Sequence`, `int` and `float`')


def read_file(filename, encoding=None):
    """Load list/tabular data from the file.

    Supported file types: 'txt', 'npy', '' (binary encoded file).
    """
    if filename.split(".")[-1] == "txt":
        fp = open(filename, 'r', encoding=encoding)
        lines = fp.readlines()
    elif filename.split(".")[-1] == "npy":
        lines = np.load(filename)
    else:
        fp = open(filename, 'rb')
        lines = pickle.load(fp)
    assert isinstance(lines, list) and len(lines) > 0
    return lines
