# coding: utf-8

import cv2
import os
import shutil
from collections import namedtuple
from copy import copy, deepcopy

from legacy_supervisely_lib import logger
from legacy_supervisely_lib.figure.fig_classes import FigClasses
from legacy_supervisely_lib.project import annotation as sly_annotation
from legacy_supervisely_lib.project.project_structure import ProjectFS
from legacy_supervisely_lib.tasks import progress_counter
from legacy_supervisely_lib.tasks import task_helpers
from legacy_supervisely_lib.tasks.task_paths import TaskPaths
from legacy_supervisely_lib.utils import config_readers
from legacy_supervisely_lib.utils import json_utils
from legacy_supervisely_lib.utils import os_utils
from legacy_supervisely_lib.utils.inference_modes import InferenceFeederFactory
from legacy_supervisely_lib.utils.json_utils import AlwaysPassingValidator, JsonConfigRW

from legacy_supervisely_lib.worker_api.rpc_servicer import SingleImageApplier
from legacy_supervisely_lib.tasks.progress_counter import progress_counter_build_model


# TODO@ Make this data structure a proper Figure and use in a uniform way.
PixelwiseClassProbasWithMapping = namedtuple(
    'PixelwiseClassProbasWithMapping', ['probas', 'class_title_to_idx_out_mapping'])
PixelwiseClassProbasWithMapping.__new__.__defaults__ = (None,) * len(PixelwiseClassProbasWithMapping._fields)

InferenceResult = namedtuple('InferenceResult', ['figures', 'pixelwise_class_probas'])
InferenceResult.__new__.__defaults__ = (None,) * len(InferenceResult._fields)


class SingleImageInferenceApplier(SingleImageApplier):
    def __init__(self, settings=None):
        logger.info('Starting base single image inference applier init.')
        settings = settings or {}
        self.settings = config_readers.update_recursively(deepcopy(self.get_default_settings()), settings)
        self.paths = TaskPaths(determine_in_project=False)
        self._load_train_config()
        self._construct_and_fill_model()
        logger.info('Base single image inference applier init done.')

    # TODO@ get rid of this layer and move handling to RPC processor?
    def apply_single_image(self, image, message):
        figures = self.inference(image, message).figures
        h, w = image.shape[:2]
        ann = sly_annotation.Annotation.new_with_objects((w, h), figures)
        return ann.pack()

    def _construct_and_fill_model(self):
        progress_dummy = progress_counter_build_model()
        progress_dummy.iter_done_report()

    def inference(self, image, message):
        raise NotImplementedError()

    @staticmethod
    def get_default_settings():
        return {}

    @property
    def class_title_to_idx_key(self):
        return 'class_title_to_idx'

    @property
    def train_classes_key(self):
        return 'classes'

    def _load_train_config(self):
        train_config_rw = JsonConfigRW(self.paths.model_config_fpath)
        if not train_config_rw.config_exists:
            raise RuntimeError('Unable to run inference, config from training wasn\'t found.')
        self.train_config = train_config_rw.load()

        self.class_title_to_idx = self.train_config[self.class_title_to_idx_key]
        self.train_classes = FigClasses(self.train_config[self.train_classes_key])
        logger.info('Read model internal class mapping', extra={'class_mapping': self.class_title_to_idx})
        logger.info('Read model out classes', extra={'classes': self.train_classes.py_container})

        # Make a separate [class title] --> [index] map that excludes the 'special' classes that should not be in the`
        # final output.
        train_class_titles = set(train_class['title'] for train_class in self.train_classes)
        self.out_class_mapping = {title: idx for title, idx in self.class_title_to_idx.items() if
                                  title in train_class_titles}

    def _determine_model_input_size(self):
        src_size = self.train_config['settings']['input_size']
        self.input_size_wh = (src_size['width'], src_size['height'])
        logger.info('Model input size is read (for auto-rescale).', extra={'input_size': {
            'width': self.input_size_wh[0], 'height': self.input_size_wh[1]
        }})


def get_default_settings_for_modes():
    return {
        'full_image': {
            'source': 'full_image',
        },
        'roi': {
            'source': 'roi',
            'bounds': {
                'left': '0px',
                'top': '0px',
                'right': '0px',
                'bottom': '0px',
            },
            'save': False,
            'class_name': 'inference_roi'
        },
        'bboxes': {
            'source': 'bboxes',
            'from_classes': '__all__',
            'padding': {
                'left': '0px',
                'top': '0px',
                'right': '0px',
                'bottom': '0px',
            },
            'save': False,
            'add_suffix': '_input_bbox'
        },
        'sliding_window': {
            'source': 'sliding_window',
            'window': {
                'width': 128,
                'height': 128,
            },
            'min_overlap': {
                'x': 0,
                'y': 0,
            },
            'save': False,
            'class_name': 'sliding_window',
        },
    }

class InferenceApplier:

    def __init__(self, single_image_inference: SingleImageInferenceApplier, default_mode_settings,
                 default_settings_for_modes=None, settings_validator_cls=AlwaysPassingValidator):
        self.single_image_inference = single_image_inference
        self.default_settings = config_readers.update_recursively(deepcopy(default_mode_settings),
                                                                  single_image_inference.get_default_settings())
        self.default_settings_for_modes = deepcopy(default_settings_for_modes or get_default_settings_for_modes())
        self.settings_validator_cls = settings_validator_cls

        self.helper = task_helpers.TaskHelperInference()
        self._determine_settings()
        self._determine_input_data()
        logger.info('Dataset inference preparation done.')


    def _determine_input_data(self):
        project_fs = ProjectFS.from_disk_dir_project(self.helper.paths.project_dir)
        logger.info('Project structure has been read. Samples: {}.'.format(project_fs.pr_structure.image_cnt))
        self.in_project_fs = project_fs

    def _determine_settings(self):
        input_config = self.helper.task_settings
        logger.info('Input config', extra={'config': input_config})

        self.config = deepcopy(self.default_settings)
        if 'mode' in input_config and 'source' in input_config['mode']:
            mode_name = input_config['mode']['source']
            default_mode_settings = self.default_settings_for_modes.get(mode_name, None)
            if default_mode_settings is not None:
                config_readers.update_recursively(self.config['mode'], deepcopy(default_mode_settings))

                config_readers.update_recursively(self.config, input_config)
        logger.info('Full config', extra={'config': self.config})
        self.settings_validator_cls.validate_inference_cfg(self.config)

        self.debug_copy_images = os.getenv('DEBUG_COPY_IMAGES') is not None

    def run_inference(self):
        out_project_fs = copy(self.in_project_fs)
        out_project_fs.root_path = self.helper.paths.results_dir
        out_project_fs.make_dirs()

        inference_feeder = InferenceFeederFactory.create(
            self.config, self.helper.in_project_meta, self.single_image_inference.train_classes)

        out_pr_meta = inference_feeder.out_meta
        out_pr_meta.to_dir(out_project_fs.project_path)

        ia_cnt = out_project_fs.pr_structure.image_cnt
        progress = progress_counter.progress_counter_inference(cnt_imgs=ia_cnt)

        for sample in self.in_project_fs:
            logger.trace('Will process image',
                         extra={'dataset_name': sample.ds_name, 'image_name': sample.image_name})
            ann_packed = json_utils.json_load(sample.ann_path)
            ann = sly_annotation.Annotation.from_packed(ann_packed, self.helper.in_project_meta)

            img = cv2.imread(sample.img_path)[:, :, ::-1]
            res_ann = inference_feeder.feed(img, ann, self.single_image_inference.inference)

            out_ann_fpath = out_project_fs.ann_path(sample.ds_name, sample.image_name)
            res_ann_packed = res_ann.pack()
            json_utils.json_dump(res_ann_packed, out_ann_fpath)

            if self.debug_copy_images:
                out_img_fpath = out_project_fs.img_path(sample.ds_name, sample.image_name)
                os_utils.ensure_base_path(out_img_fpath)
                shutil.copy(sample.img_path, out_img_fpath)

            progress.iter_done_report()

        progress_counter.report_inference_finished()
