# coding: utf-8

import os

import supervisely_lib as sly
from supervisely_lib.metric.precision_recall_metric import PrecisionRecallMetric
from supervisely_lib.metric.iou_metric import IOU
from supervisely_lib.metric.common import check_class_mapping, CLASSES_MAPPING
from supervisely_lib.io.json import load_json_file


def main():
    settings = load_json_file(sly.TaskPaths.TASK_CONFIG_PATH)
    sly.logger.info('Input settings:', extra={'config': settings})

    if IOU not in settings:
        raise RuntimeError('"{}" field is missing. Please set Intersection over Union threshold'.format(IOU))

    metric = PrecisionRecallMetric(settings[CLASSES_MAPPING], settings[IOU])
    applier = sly.MetricProjectsApplier(metric, settings)
    check_class_mapping(applier.project_gt, applier.project_pred, settings[CLASSES_MAPPING])
    applier.run_evaluation()
    metric.log_total_metrics()


if __name__ == '__main__':
    if os.getenv('DEBUG_LOG_TO_FILE', None):
        sly.add_default_logging_into_file(sly.logger, sly.TaskPaths.DEBUG_DIR)
    sly.main_wrapper('METRIC_EVALUATION', main)
