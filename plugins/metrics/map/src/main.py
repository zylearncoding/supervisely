# coding: utf-8

import os

import supervisely_lib as sly
from supervisely_lib.metric.map_metric import MAPMetric
from supervisely_lib.metric.common import check_class_mapping, CLASSES_MAPPING
from supervisely_lib.io.json import load_json_file
from supervisely_lib.metric.iou_metric import IOU


CONFIDENCE_TAG_NAME = 'confidence_tag_name'


def main():
    settings = load_json_file(sly.TaskPaths.TASK_CONFIG_PATH)
    sly.logger.info('Input settings:', extra={'config': settings})

    if IOU not in settings:
        raise RuntimeError('"{}" field is missing. Please set Intersection over Union threshold'.format(IOU))
    if CONFIDENCE_TAG_NAME not in settings:
        raise RuntimeError(
            f'{CONFIDENCE_TAG_NAME!r} field is missing. Please set the tag name to read prediction confidence from.')

    confidence_tag_name = settings[CONFIDENCE_TAG_NAME]
    metric = MAPMetric(settings[CLASSES_MAPPING], settings[IOU], confidence_tag_name=confidence_tag_name)
    applier = sly.MetricProjectsApplier(metric, settings)

    # Input sanity checks.
    check_class_mapping(applier.project_gt, applier.project_pred, settings[CLASSES_MAPPING])
    if not applier.project_pred.meta.tag_metas.has_key(confidence_tag_name):
        raise RuntimeError(f'Tag {confidence_tag_name!r} cannot be found in the project with predictions '
                           f'{applier.project_pred.name!r} does not have that tag. Make sure you specify the correct '
                           f'confidence tag name as a {CONFIDENCE_TAG_NAME!r} setting in the plugin config.')

    applier.run_evaluation()
    metric.log_total_metrics()


if __name__ == '__main__':
    if os.getenv('DEBUG_LOG_TO_FILE', None):
        sly.add_default_logging_into_file(sly.logger, sly.TaskPaths.DEBUG_DIR)
    sly.main_wrapper('METRIC_EVALUATION', main)
