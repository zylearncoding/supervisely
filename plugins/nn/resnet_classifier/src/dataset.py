# coding: utf-8
import os
import torch
from torchvision.transforms import ToTensor, Normalize, Compose
from supervisely_lib.nn.dataset import SlyDataset

import supervisely_lib as sly


input_image_normalizer = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ResnetDataset(SlyDataset):

    def __init__(self, project_meta, samples, out_size, class_mapping, out_classes, allow_corrupted_cnt, spec_tags: list):
        super().__init__(project_meta=project_meta,
                         samples=samples,
                         out_size=out_size,
                         class_mapping=class_mapping,
                         bkg_color=None,
                         allow_corrupted_cnt=allow_corrupted_cnt)
        self.out_classes = out_classes
        self.spec_tags = set(spec_tags)

    def _get_sample_impl(self, img_fpath, ann_fpath):
        img = sly.image.read(img_fpath)
        img = sly.image.resize(img, self._out_size)
        ann = self.load_annotation(ann_fpath)

        img_tags = set(tag.name for tag in ann.img_tags if tag.value is None)
        img_tags -= self.spec_tags
        img_tags &= set(self.out_classes)
        img_tags = sorted(img_tags)

        if len(img_tags) != 1:
            raise RuntimeError("Expected exactly one classification tag per image. "
                               "Instead got {} tags: {} for image {}.".
                               format(len(img_tags), str(img_tags), os.path.basename(img_fpath)))

        cls_title = img_tags[0]

        gt_id = self._class_mapping[cls_title]
        return input_image_normalizer(img), torch.tensor(gt_id)
