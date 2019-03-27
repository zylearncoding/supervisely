# coding: utf-8
import os
from copy import deepcopy

import numpy as np
from legacy_supervisely_lib.project.annotation import Annotation
from legacy_supervisely_lib.project.project_structure import ProjectFS
from legacy_supervisely_lib.project.project_meta import ProjectMeta
from legacy_supervisely_lib.tasks import progress_counter
from legacy_supervisely_lib.tasks import task_helpers
from legacy_supervisely_lib.tasks import task_paths
from legacy_supervisely_lib.utils import config_readers
from legacy_supervisely_lib.utils import json_utils
from legacy_supervisely_lib.utils import logging_utils
from legacy_supervisely_lib import logger
from legacy_supervisely_lib import sly_logger


def progress_counter_metric_evaluation(cnt_imgs, ext_logger=None):
    ctr = progress_counter.ProgressCounter('metric evaluation', cnt_imgs, ext_logger=ext_logger)
    return ctr


class MetricEvaluator:
    @staticmethod
    def _check_samples_names_sameness(sample_1, sample_2):
        if sample_1.ds_name != sample_2.ds_name or sample_1.image_name != sample_2.image_name:
            raise RuntimeError("Input projects should contain same images. Image {} in dataset {} does not exist"
                               " in second project".format(sample_1.ds_name, sample_1.image_name))

    def _determine_settings(self):
        input_config = self.helper.task_settings
        logger.info('Input config', extra={'config': input_config})

        config = deepcopy(self.default_settings)

        config_readers.update_recursively(config, input_config)
        logger.info('Full config', extra={'config': config})
        if len(config['classes_mapping']) < 1:
            raise RuntimeError('At least one classes pair should be defined')
        self.classes_mapping = config['classes_mapping']
        self.config = config

    def _check_project_meta(self):
        self.metric_res = {}
        for k, v in self.classes_mapping.items():
            if self.in_pr_meta_1.classes[k] is None:
                raise RuntimeError('Class {} does not exist in input project'.format(k))
            if self.in_pr_meta_2.classes[v] is None:
                raise RuntimeError('Class {} does not exist in input project'.format(v))
            self.metric_res[k + ':' + v] = []

    def _determine_input_data(self):
        project_1 = self.config['project_1']
        project_2 = self.config.get('project_2')
        if project_2 is None:
            project_2 = project_1

        data_dir = os.path.join(self.helper.paths.task_dir, 'data')
        for pr in [project_1, project_2]:
            if not os.path.exists(os.path.join(data_dir, pr)):
                raise RuntimeError('Project {} does not exist.'.format(pr))

        self.in_project_fs_1 = ProjectFS.from_disk(data_dir, project_1)
        self.in_project_fs_2 = ProjectFS.from_disk(data_dir, project_2)
        if self.in_project_fs_1.image_cnt != self.in_project_fs_2.image_cnt:
            raise RuntimeError('Projects should contain same number of samples.')
        logger.info('Projects structure has been read. Samples: {}.'.format(self.in_project_fs_1.image_cnt))

        self.in_pr_meta_1 = ProjectMeta.from_dir(os.path.join(data_dir, project_1))
        self.in_pr_meta_2 = ProjectMeta.from_dir(os.path.join(data_dir, project_2))

    def __init__(self, default_settings={}):
        logger.info('Will init all required to evaluation.')
        self.helper = task_helpers.TaskHelperMetrics()

        self.default_settings = deepcopy(default_settings)
        self._determine_settings()
        self._determine_input_data()
        self._check_project_meta()

    def _evaluate_on_img(self, ann_1, ann_2):
        raise NotImplementedError

    def run_evaluation(self):
        samples_1 = list(self.in_project_fs_1)
        samples_1.sort(key=lambda x: (x.ds_name, x.image_name))
        samples_2 = list(self.in_project_fs_2)
        samples_2.sort(key=lambda x: (x.ds_name, x.image_name))

        ia_cnt = self.in_project_fs_1.pr_structure.image_cnt
        progress = progress_counter_metric_evaluation(cnt_imgs=ia_cnt)

        for sample_1, sample_2 in zip(samples_1, samples_2):
            logger.trace('Will process image',
                         extra={'dataset_name': sample_1.ds_name, 'image_name': sample_1.image_name})
            self._check_samples_names_sameness(sample_1, sample_2)

            ann_packed = json_utils.json_load(sample_1.ann_path)
            ann_1 = Annotation.from_packed(ann_packed, self.in_pr_meta_1)
            ann_packed = json_utils.json_load(sample_2.ann_path)
            ann_2 = Annotation.from_packed(ann_packed, self.in_pr_meta_2)
            self._evaluate_on_img(ann_1, ann_2)
            progress.iter_done_report()

    def log_result_metrics(self):
        raise NotImplementedError
