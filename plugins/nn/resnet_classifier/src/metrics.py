# coding: utf-8

import numpy as np
import torch.nn.functional as F


class Accuracy:  # multiclass
    def __init__(self, ignore_index=None):
        self.ignore_index = ignore_index

    def __call__(self, outputs, targets):
        outputs = F.softmax(outputs, dim=1)
        outputs = outputs.data.cpu().numpy()
        outputs = np.argmax(outputs, axis=1)

        targets = targets.data.cpu().numpy()

        total_preds = np.prod(outputs.shape)
        if self.ignore_index is not None:
            total_preds -= np.sum((targets == self.ignore_index).astype(int))
        correct_preds = np.sum((outputs == targets).astype(int))
        res = correct_preds / total_preds
        return res

