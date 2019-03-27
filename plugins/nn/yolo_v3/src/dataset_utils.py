# coding: utf-8

import numpy as np

import supervisely_lib as sly


def load_ann(ann_fpath, classes_mapping, project_meta):
    ann = sly.Annotation.load_json_file(ann_fpath, project_meta)
    (h, w) = ann.img_size

    gt_boxes, classes_text, classes = [], [], []

    for label in ann.labels:
        gt_index = classes_mapping.get(label.obj_class.name, None)
        if gt_index is None:
            raise RuntimeError('Missing class mapping (title to index). Class {}.'.format(label.obj_class.name))
        rect = label.geometry.to_bbox()
        x = rect.center.col / w
        y = rect.center.row / h
        r_width = rect.width / w
        r_height = rect.height / h
        gt_boxes.extend([gt_index, x, y, r_width, r_height])
    num_boxes = len(ann.labels)
    return num_boxes, gt_boxes


def load_dataset(samples, classes_mapping, project_meta):
    img_paths = []
    gts = []
    num_boxes = []
    for img_path, ann_path in samples:
        img_paths.append(img_path.encode('utf-8'))
        num_b, gt_boxes = load_ann(ann_path, classes_mapping, project_meta)
        gts.append(gt_boxes)
        num_boxes.append(num_b)

    gts = [np.array(x).astype(np.float32).tolist() for x in gts]

    return img_paths, gts, num_boxes
