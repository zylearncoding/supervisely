# coding: utf-8

import os
from os.path import join

import supervisely_lib as sly
from supervisely_lib.nn import config as sly_nn_config
from darknet_utils import load_net, detect_pyimage


def class_to_idx_config_key():
    return 'class_title_to_idx'

  
def train_classes_key():
    return 'out_classes'


class YoloJsonConfigValidator(sly_nn_config.JsonConfigValidator):
    def validate_train_cfg(self, config):
        super().validate_train_cfg(config)
        sp_classes = config['special_classes']
        if len(set(sp_classes.values())) != len(sp_classes):
            raise RuntimeError('Non-unique special classes in train config.')


def yolo_preds_to_sly_rects(detections, idx_to_class, confidence_tag_meta):
    labels = []
    for classId, confidence, box in detections:
            xmin = box[0] - box[2] / 2
            ymin = box[1] - box[3] / 2
            xmax = box[0] + box[2] / 2
            ymax = box[1] + box[3] / 2
            rect = sly.Rectangle(round(ymin), round(xmin), round(ymax), round(xmax))

            label = sly.Label(rect, idx_to_class[classId])

            confidence_tag = sly.Tag(confidence_tag_meta, value=round(float(confidence), 4))
            label = label.add_tag(confidence_tag)
            labels.append(label)
    return labels


def construct_model(model_dir):
    src_train_cfg_path = join(model_dir, 'model.cfg')
    with open(src_train_cfg_path) as f:
        src_config = f.readlines()

    def repl_batch(row):
        if 'batch=' in row:
            return 'batch=1\n'
        if 'subdivisions=' in row:
            return 'subdivisions=1\n'
        return row

    changed_config = [repl_batch(x) for x in src_config]

    inf_cfg_path = join(model_dir, 'inf_model.cfg')
    if not os.path.exists(inf_cfg_path):
        with open(inf_cfg_path, 'w') as f:
            f.writelines(changed_config)

    model = load_net(inf_cfg_path.encode('utf-8'),
                        join(model_dir, 'model.weights').encode('utf-8'),
                        0)
    return model


def infer_on_image(image, model, idx_to_class, confidence_thresh, confidence_tag_meta, num_classes):
    detections_result = detect_pyimage(model, num_classes, image, thresh=confidence_thresh)
    return yolo_preds_to_sly_rects(detections_result, idx_to_class, confidence_tag_meta)
