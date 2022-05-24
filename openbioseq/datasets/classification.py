import torch

from openbioseq.models.utils import precision_recall_f1, support
from openbioseq.utils import print_log

from .registry import DATASETS
from .base import BaseDataset
from .utils import to_numpy


@DATASETS.register_module
class ClassificationDataset(BaseDataset):
    """The dataset outputs one view of an image, containing some other
        information such as label, idx, etc.

    Args:
        data_source (dict): Data source defined in
            `mmselfsup.datasets.data_sources`.
        pipeline (list[dict]): A list of dict, where each element represents
            an operation defined in `mmselfsup.datasets.pipelines`.
        prefetch (bool, optional): Whether to prefetch data. Defaults to False.
    """

    def __init__(self, data_source, pipeline, prefetch=False):
        super(ClassificationDataset, self).__init__(data_source, pipeline, prefetch)

    def __getitem__(self, idx):
        data, target = self.data_source.get_sample(idx)
        data = self.pipeline(data)
        if self.prefetch:
            data = torch.from_numpy(to_numpy(data))
        return dict(data=data, gt_label=target, idx=idx)

    def evaluate(self,
                 scores, keyword, logger=None,
                 metric='accuracy',
                 metric_options=None,
                 indices=None):
        """The evaluation function to output accuracy.

        Args:
            scores (tensor): Prediction values.
            keyword (str): The corresponding head name.
            logger (logging.Logger | str | None, optional): The defined logger
                to be used. Defaults to None.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'topk', 'thrs' and 'average_mode'.
                Defaults to None.
        Returns:
            dict: evaluation results
        """
        if metric_options is None:
            metric_options = {'topk': (1, 5)}
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        eval_res = {}
        eval_log = []
        allowed_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'support',]
        average_mode = metric_options.get('average_mode', 'macro')
        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metric {invalid_metrics} is not supported.')
        
        target = torch.LongTensor(self.data_source.labels)
        if indices is not None:
            target = target[indices]
        assert scores.size(0) == target.size(0), \
            "Inconsistent length for results and labels, {} vs {}".format(
            scores.size(0), target.size(0))
        
        if 'accuracy' in metrics:
            topk = metric_options.get('topk', (1, 5))
            _, pred = scores.topk(max(topk), dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))  # KxN
            for k in topk:
                correct_k = correct[:k].contiguous().view(-1).float().sum(0).item()
                acc = correct_k * 100.0 / scores.size(0)
                eval_res[f"{keyword}_top{k}"] = acc
                eval_log.append("{}_top{}: {:.03f}".format(keyword, k, acc))
        
        if 'support' in metrics:
            support_value = support(scores, target, average_mode=average_mode)
            eval_res[f'{keyword}_support'] = support_value
            eval_log.append("{}_support: {:.03f}".format(keyword, support_value))

        precision_recall_f1_keys = ['precision', 'recall', 'f1_score']
        if len(set(metrics) & set(precision_recall_f1_keys)) != 0:
            thrs = metric_options.get('thrs', 0.)
            if thrs is not None:
                precision_recall_f1_values = precision_recall_f1(
                    scores, target, average_mode=average_mode, thrs=thrs)
            else:
                precision_recall_f1_values = precision_recall_f1(
                    scores, target, average_mode=average_mode)
            for key, values in zip(precision_recall_f1_keys,
                                   precision_recall_f1_values):
                if key in metrics:
                    if isinstance(thrs, tuple):
                        eval_res.update({f'{key}_thr_{thr:.2f}': value
                            for thr, value in zip(thrs, values)
                        })
                    else:
                        eval_res[key] = values
                        eval_log.append("{}_{}: {:.03f}".format(keyword, key, values))
        
        if logger is not None and logger != 'silent':
            for _log in eval_log:
                print_log(_log, logger=logger)
        
        return eval_res
