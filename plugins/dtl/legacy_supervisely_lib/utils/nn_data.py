# coding: utf-8

from collections import defaultdict
from threading import Lock

import numpy as np

from ..project.annotation import Annotation
from ..figure.fig_classes import FigClasses
from ..figure.figure_bitmap import FigureBitmap
from ..figure.figure_rectangle import FigureRectangle
from ..figure.rectangle import Rect
from ..sly_logger import logger
from .json_utils import json_load, JsonConfigRW


def samples_by_tags(tags, project_fs, project_meta):
    samples = defaultdict(list)
    for item_descr in project_fs:
        ann_packed = json_load(item_descr.ann_path)
        ann = Annotation.from_packed(ann_packed, project_meta)
        for req_tag in tags:
            if (req_tag == '__all__') or (req_tag in ann.tags):
                item_descr.ia_data['obj_cnt'] = len(ann['objects'])
                samples[req_tag].append(item_descr)

    return samples


def ensure_samples_nonempty(samples_lst, tag_name):
    if len(samples_lst) < 1:
        raise RuntimeError('There are no annotations with tag "{}"'.format(tag_name))
    if sum(x.ia_data['obj_cnt'] for x in samples_lst) == 0:
        raise RuntimeError('There are no objects in annotations with tag "{}"'.format(tag_name))


class CorruptedSampleCatcher(object):
    def __init__(self, allow_corrupted_cnt):
        self.fails_allowed = allow_corrupted_cnt
        self._failed_uids = set()
        self._lock = Lock()

    def exec(self, uid, log_dct, f, *args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            self._lock.acquire()
            if uid not in self._failed_uids:
                self._failed_uids.add(uid)
                logger.warn('Sample processing error.', exc_info=True, extra={**log_dct, 'exc_str': str(e)})
            fail_cnt = len(self._failed_uids)
            self._lock.release()

            if fail_cnt > self.fails_allowed:
                raise RuntimeError('Too many errors occurred while processing samples. '
                                   'Allowed: {}.'.format(self.fails_allowed))


def create_segmentation_classes(in_project_classes, special_classes_config, bkg_input_idx,
                                weights_init_type, model_config_fpath, class_to_idx_config_key, start_class_id=1):
    extra_classes = {}
    special_class_ids = {}
    bkg_title = special_classes_config.get('background', None)
    if bkg_title is not None:
        extra_classes = {bkg_title: '#222222'}
        special_class_ids = {bkg_title: bkg_input_idx}

    exclude_titles = []
    neutral_title = special_classes_config.get('neutral', None)
    if neutral_title is not None:
        exclude_titles.append(neutral_title)
    out_classes = make_out_classes(in_project_classes, shape='bitmap', exclude_titles=exclude_titles,
                                   extra_classes=extra_classes)
    logger.info('Determined model out classes', extra={'out_classes': out_classes.py_container})
    in_project_class_to_idx = make_new_class_to_idx_map(in_project_classes, start_class_id=start_class_id,
                                                        preset_class_ids=special_class_ids,
                                                        exclude_titles=exclude_titles)
    class_title_to_idx = infer_training_class_to_idx_map(weights_init_type,
                                                         in_project_class_to_idx,
                                                         model_config_fpath,
                                                         class_to_idx_config_key,
                                                         special_class_ids=special_class_ids)
    logger.info('Determined class mapping.', extra={'class_mapping': class_title_to_idx})
    return out_classes, class_title_to_idx


def make_new_class_to_idx_map(in_project_classes, start_class_id, preset_class_ids=None, exclude_titles=None):
    preset_class_ids = preset_class_ids or dict()
    exclude_titles = exclude_titles or []
    if any(title in preset_class_ids for title in exclude_titles):
        raise RuntimeError(
            'Unable to construct class name to integer id map: preset classes names overlap with excluded classes.')
    sorted_titles = sorted((x['title'] for x in in_project_classes))
    if len(sorted_titles) != len(set(sorted_titles)):
        raise RuntimeError('Unable to construct class name to integer id map: class names are not unique.')

    class_title_to_idx = preset_class_ids.copy()
    next_title_id = start_class_id
    already_used_ids = set(class_title_to_idx.values())
    for title in sorted_titles:
        if title not in class_title_to_idx and title not in exclude_titles:
            while next_title_id in already_used_ids:
                next_title_id += 1
            class_title_to_idx[title] = next_title_id
            already_used_ids.add(next_title_id)
    return class_title_to_idx


def make_out_classes(in_project_classes, shape, exclude_titles=None, extra_classes=None):
    exclude_titles = exclude_titles or []
    extra_classes = extra_classes or dict()

    if any(title in extra_classes for title in exclude_titles):
        raise RuntimeError(
            'Unable to construct class name to integer id map: extra classes names overlap with excluded classes.')

    out_classes = FigClasses()
    all_used_titles = set()
    for in_class in in_project_classes:
        title = in_class['title']
        if title not in exclude_titles:
            out_classes.add({
                'title': title,
                'shape': shape,
                'color': in_class['color'],
            })
            all_used_titles.add(title)
    for title, color in extra_classes.items():
        if title not in all_used_titles:
            out_classes.add({
                'title': title,
                'shape': shape,
                'color': color,
            })
            all_used_titles.add(title)
    return out_classes


def read_validate_model_class_to_idx_map(model_config_fpath, in_project_classes_set, class_to_idx_config_key):
    """Reads class id --> int index mapping from the model config; checks that the set of classes matches the input."""

    model_config_rw = JsonConfigRW(model_config_fpath)
    if not model_config_rw.config_exists:
        raise RuntimeError('Unable to continue_training, config for previous training wasn\'t found.')
    model_config = model_config_rw.load()

    model_class_mapping = model_config.get(class_to_idx_config_key, None)
    if model_class_mapping is None:
        raise RuntimeError('Unable to continue_training, model does not have class mapping information.')
    model_classes_set = set(model_class_mapping.keys())

    if model_classes_set != in_project_classes_set:
        error_message_text = 'Unable to continue_training, sets of classes for model and dataset do not match.'
        logger.critical(
            error_message_text, extra={'model_classes': model_classes_set, 'dataset_classes': in_project_classes_set})
        raise RuntimeError(error_message_text)
    return model_class_mapping.copy()


def infer_training_class_to_idx_map(weights_init_type, in_project_class_to_idx, model_config_fpath,
                                    class_to_idx_config_key, special_class_ids=None):
    if weights_init_type == 'transfer_learning':
        logger.info('Transfer learning mode, using a class mapping created from scratch.')
        class_title_to_idx = in_project_class_to_idx
    elif weights_init_type == 'continue_training':
        logger.info('Continued training mode, reusing the existing class mapping from the model.')
        class_title_to_idx = read_validate_model_class_to_idx_map(
            model_config_fpath=model_config_fpath,
            in_project_classes_set=set(in_project_class_to_idx.keys()),
            class_to_idx_config_key=class_to_idx_config_key)
    else:
        raise RuntimeError('Unknown weights init type: {}'.format(weights_init_type))

    if special_class_ids is not None:
        for class_title, requested_class_id in special_class_ids.items():
            effective_class_id = class_title_to_idx[class_title]
            if requested_class_id != effective_class_id:
                error_msg = ('Unable to start training. Effective integer id for class {} does not match the ' +
                             'requested value in the training config ({} vs {}).'.format(
                                 class_title, effective_class_id, requested_class_id))
                logger.critical(error_msg, extra={'class_title_to_idx': class_title_to_idx,
                                                  'special_class_ids': special_class_ids})
                raise RuntimeError(error_msg)
    return class_title_to_idx


# converts predictions (encoded as numbers)
def prediction_to_sly_bitmaps(class_title_to_idx, pred):
    size_wh = (pred.shape[1], pred.shape[0])
    out_figures = []
    for cls_title in sorted(class_title_to_idx.keys()):
        cls_idx = class_title_to_idx[cls_title]
        class_pred_mask = pred == cls_idx
        new_objs = FigureBitmap.from_mask(cls_title, size_wh, origin=(0, 0), mask=class_pred_mask)
        out_figures.extend(new_objs)
    return out_figures


# converts tf_model inference output
def detection_preds_to_sly_rects(inverse_mapping, net_out, img_shape, min_score_thresold):
    img_wh = img_shape[1::-1]
    (boxes, scores, classes, num) = net_out
    out_figures = []
    thr_mask = np.squeeze(scores) > min_score_thresold
    for box, class_id, score in zip(np.squeeze(boxes)[thr_mask],
                                    np.squeeze(classes)[thr_mask],
                                    np.squeeze(scores)[thr_mask]):

            xmin = int(box[1] * img_shape[1])
            ymin = int(box[0] * img_shape[0])
            xmax = int(box[3] * img_shape[1])
            ymax = int(box[2] * img_shape[0])
            cls_name = inverse_mapping[int(class_id)]
            rect = Rect(xmin, ymin, xmax, ymax)
            new_objs = FigureRectangle.from_rect(cls_name, img_wh, rect)
            for x in new_objs:
                x.data['score'] = float(score)
            out_figures.extend(new_objs)
    return out_figures
