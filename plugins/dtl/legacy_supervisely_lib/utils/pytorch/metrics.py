# coding: utf-8
import numpy as np


class MultiClassAccuracy:
    def __init__(self, ignore_index=None, squeeze_targets=True):
        self._ignore_index = ignore_index
        self._squeeze_targets = squeeze_targets

    def __call__(self, outputs, targets):
        outputs = outputs.data.cpu().numpy()
        outputs = np.argmax(outputs, axis=1)

        targets = targets.data.cpu().numpy()
        if self._squeeze_targets:
            targets = np.squeeze(targets, 1)  # 4d to 3d

        total_pixels = np.prod(outputs.shape)
        if self._ignore_index is not None:
            total_pixels -= np.sum((targets == self._ignore_index).astype(int))
        correct_pixels = np.sum((outputs == targets).astype(int))
        res = correct_pixels / total_pixels
        return res
