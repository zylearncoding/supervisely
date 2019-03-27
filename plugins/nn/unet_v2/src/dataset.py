# coding: utf-8
import numpy as np

from torchvision.transforms import ToTensor, Normalize, Compose
from supervisely_lib.nn.dataset import SlyDataset


input_image_normalizer = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class UnetV2Dataset(SlyDataset):
    def _get_sample_impl(self, img_fpath, ann_fpath):
        img, gt = super()._get_sample_impl(img_fpath=img_fpath, ann_fpath=ann_fpath)
        gt = np.expand_dims(gt, 0)
        return input_image_normalizer(img), gt
