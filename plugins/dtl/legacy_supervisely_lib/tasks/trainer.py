# coding: utf-8

import os
import os.path
from copy import deepcopy

from legacy_supervisely_lib import logger
from legacy_supervisely_lib.project.project_structure import ProjectFS
from legacy_supervisely_lib.tasks import task_helpers
from legacy_supervisely_lib.utils import config_readers
from legacy_supervisely_lib.utils import nn_data
from legacy_supervisely_lib.utils import os_utils
from legacy_supervisely_lib.utils.json_utils import JsonConfigRW, SettingsValidator


# float progress of training, since zero
def epoch_float(epoch, train_it, train_its):
    res = epoch + train_it / float(train_its)
    return res


class SuperviselyModelTrainer:

    def __init__(self, default_settings):
        logger.info('Will init all required to train.')
        self.helper = task_helpers.TaskHelperTrain()

        self.default_settings = deepcopy(default_settings)
        self._determine_settings()
        self._determine_model_classes()
        self._determine_out_config()
        self._construct_samples_dct()
        self._construct_data_loaders()
        self._construct_and_fill_model()
        self._construct_loss()

        self.epoch_flt = 0

    @property
    def class_title_to_idx_key(self):
        return 'class_title_to_idx'

    @property
    def train_classes_key(self):
        return 'classes'

    def get_start_class_id(self):
        # Returns the first integer id to use when assigning integer ids to class names.
        # The usual setting for segmentation network, where background is often a special class with id 0.
        return 1

    def _determine_settings(self):
        input_config = self.helper.task_settings
        logger.info('Input config', extra={'config': input_config})
        config = deepcopy(self.default_settings)
        config_readers.update_recursively(config, input_config)
        logger.info('Full config', extra={'config': config})
        SettingsValidator.validate_train_cfg(config)
        self.config = config

    def _determine_model_classes_segmentation(self, bkg_input_idx):
        self.out_classes, self.class_title_to_idx = nn_data.create_segmentation_classes(
            in_project_classes=self.helper.in_project_meta.classes,
            special_classes_config=self.config.get('special_classes', {}),
            bkg_input_idx=bkg_input_idx,
            weights_init_type=self.config['weights_init_type'],
            model_config_fpath=self.helper.paths.model_config_fpath,
            class_to_idx_config_key=self.class_title_to_idx_key,
            start_class_id=(1 if bkg_input_idx == 0 else 0))

    def _determine_model_classes_detection(self):
        in_project_classes = self.helper.in_project_meta.classes
        self.out_classes = nn_data.make_out_classes(in_project_classes, shape='rectangle')
        logger.info('Determined model out classes', extra={'classes': self.out_classes.py_container})
        in_project_class_to_idx = nn_data.make_new_class_to_idx_map(in_project_classes,
                                                                    start_class_id=self.get_start_class_id())
        self.class_title_to_idx = nn_data.infer_training_class_to_idx_map(
            self.config['weights_init_type'],
            in_project_class_to_idx,
            self.helper.paths.model_config_fpath,
            class_to_idx_config_key=self.class_title_to_idx_key)
        logger.info('Determined class mapping.', extra={'class_mapping': self.class_title_to_idx})

    def _determine_model_classes(self):
        # Use _determine_model_classes_segmentation() or _determine_model_classes_detection() here depending on the
        # model needs.
        raise NotImplementedError()

    def _determine_out_config(self):
        self.out_config = {
            'settings': self.config,
            self.train_classes_key: self.out_classes.py_container,
            self.class_title_to_idx_key: self.class_title_to_idx,
        }

    def _construct_and_fill_model(self):
        raise NotImplementedError()

    def _construct_loss(self):
        # Useful for Tensorflow based models.
        raise NotImplementedError()

    def _construct_samples_dct(self):
        logger.info('Will collect samples (img/ann pairs).')
        self.name_to_tag = self.config['dataset_tags']
        project_fs = ProjectFS.from_disk_dir_project(self.helper.paths.project_dir)
        logger.info('Project structure has been read. Samples: {}.'.format(project_fs.pr_structure.image_cnt))
        self.samples_dct = nn_data.samples_by_tags(tags=list(self.name_to_tag.values()), project_fs=project_fs,
                                                   project_meta=self.helper.in_project_meta)

    def _construct_data_loaders(self):
        # Pipeline-specific code to set up data loading should go here.
        raise NotImplementedError()

    def _dump_model_weights(self, out_dir):
        raise NotImplementedError

    def _dump_model(self, is_best, opt_data):
        out_dir = self.helper.checkpoints_saver.get_dir_to_write()
        JsonConfigRW(os.path.join(out_dir, 'config.json')).save(self.out_config)
        self._dump_model_weights(out_dir)
        sizeb = os_utils.get_directory_size(out_dir)
        self.helper.checkpoints_saver.saved(is_best, sizeb, opt_data)

    def train(self):
        raise NotImplementedError()
