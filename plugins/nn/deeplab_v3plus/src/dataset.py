# coding: utf-8
import numpy as np
from supervisely_lib.nn.dataset import SlyDataset


class DeepLabV3Dataset(SlyDataset):

    def _get_sample_impl(self, img_fpath, ann_fpath):
        img, gt = super()._get_sample_impl(img_fpath=img_fpath, ann_fpath=ann_fpath)
        gt = np.expand_dims(gt, 2)
        return img, gt
