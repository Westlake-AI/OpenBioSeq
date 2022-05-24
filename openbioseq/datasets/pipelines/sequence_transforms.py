import numpy as np

from ..utils import to_numpy
from ..registry import PIPELINES


@PIPELINES.register_module
class RandomJitter(object):
    """ Random Seq Jittering

        https://arxiv.org/pdf/1706.00527.pdf
    """

    def __init__(self, sigma=0.8, p=0.5):
        assert 0 <= p <= 1.0, \
            f'The prob should be in range [0,1], got {p} instead.'
        self.sigma = sigma
        self.prob = p

    def __call__(self, data):
        if np.random.rand() > self.prob:
            return data
        data = to_numpy(data)
        data += np.random.normal(loc=0., scale=self.sigma, size=data.shape)
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module
class RandomScaling(object):
    """ Random Seq Scaling

        https://arxiv.org/pdf/1706.00527.pdf
    """
    def __init__(self, sigma=1.1, p=0.5):
        assert 0 <= p <= 1.0, \
            f'The prob should be in range [0,1], got {p} instead.'
        self.sigma = sigma
        self.prob = p

    def __call__(self, data):
        if np.random.rand() > self.prob:
            return data
        data = to_numpy(data)
        data = data.reshape(1, data.shape[0], -1)

        factor = np.random.normal(
            loc=2., scale=self.sigma, size=(data.shape[0], data.shape[2]))
        ai = []
        for i in range(data.shape[1]):
            xi = data[:, i, :]
            ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
        ret = np.concatenate((ai), axis=1)
        ret = ret.reshape(data.shape[1], -1)
        return ret

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module
class RandomPermutation(object):
    """ Random Seq Permutation """
    def __init__(self, max_segments=1.1, seg_mode="random", p=0.5):
        assert 0 <= p <= 1.0, \
            f'The prob should be in range [0,1], got {p} instead.'
        self.max_segments = max_segments
        self.seg_mode = seg_mode
        self.prob = p

    def __call__(self, data):
        if np.random.rand() > self.prob:
            return data
        data = to_numpy(data)
        data = data.reshape(1, data.shape[0], -1)

        orig_steps = np.arange(data.shape[2])

        num_segs = np.random.randint(1, self.max_segments, size=(data.shape[0]))

        ret = np.zeros_like(data)
        for i, pat in enumerate(data):
            if num_segs[i] > 1:
                if self.seg_mode == "random":
                    split_points = np.random.choice(data.shape[2] - 2, num_segs[i] - 1, replace=False)
                    split_points.sort()
                    splits = np.split(orig_steps, split_points)
                else:
                    splits = np.array_split(orig_steps, num_segs[i])
                warp = np.concatenate(np.random.permutation(splits)).ravel()
                ret[i] = pat[0,warp]
            else:
                ret[i] = pat
        ret = ret.reshape(data.shape[1], -1)
        return ret

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
