import os
from PIL import Image

from ..registry import DATASOURCES


@DATASOURCES.register_module
class ImageList(object):
    """The implementation for loading any image list file.

    The `ImageList` can load an annotation file or a list of files and merge
    all data records to one list. If data is unlabeled, the gt_label will be
    set -1.

    Args:
        root (str): Root of image dataset.
        list_file (str): Path to the list file.
        splitor (str): Split each seqence in the data.
        return_label (bool): Whether to return supervised labels.
    """

    CLASSES = None

    def __init__(self,
                 root,
                 list_file,
                 splitor=" ",
                 return_label=True):
        fp = open(list_file, 'r')
        lines = fp.readlines()
        assert splitor in [" ", ",", ";"]
        self.has_labels = len(lines[0].split(splitor)) == 2
        self.return_label = return_label
        if self.has_labels:
            self.fns, self.labels = zip(*[l.strip().split(splitor) for l in lines])
            self.labels = [int(l) for l in self.labels]
        else:
            # assert self.return_label is False
            self.labels = None
            self.fns = [l.strip() for l in lines]
        self.fns = [os.path.join(root, fn) for fn in self.fns]

    def get_length(self):
        return len(self.fns)

    def get_sample(self, idx):
        img = Image.open(self.fns[idx])
        img = img.convert('RGB')
        if self.has_labels and self.return_label:
            target = self.labels[idx]
            return img, target
        else:
            return img
