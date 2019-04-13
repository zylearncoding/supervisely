# coding: utf-8

import supervisely_lib as sly
from supervisely_lib.nn import config as sly_nn_config
from darknet_utils import detect_pyimage


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


def infer_on_image(image, model, idx_to_class, confidence_thresh, confidence_tag_meta, num_classes):
    detections_result = detect_pyimage(model, num_classes, image, thresh=confidence_thresh)
    return yolo_preds_to_sly_rects(detections_result, idx_to_class, confidence_tag_meta)
