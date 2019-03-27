# coding: utf-8

import os
import cv2
import numpy as np

from torch.nn import DataParallel
import supervisely_lib as sly
from supervisely_lib.annotation.annotation import Annotation
from supervisely_lib.sly_logger import logger
from supervisely_lib.task.paths import TaskPaths
from supervisely_lib.nn.hosted.constants import SETTINGS
from supervisely_lib.nn.hosted.inference_single_image import SingleImageInferenceBase, GPU_DEVICE
from supervisely_lib.nn.hosted.inference_batch import BatchInferenceApplier
from supervisely_lib.nn.hosted.inference_modes import InfModeFullImage
from supervisely_lib.nn import raw_to_labels
from supervisely_lib.nn.pytorch import inference as pytorch_inference
from supervisely_lib.nn.config import JsonConfigValidator
from supervisely_lib.nn.pytorch.weights import WeightsRW

from icnet import ICNet, make_icnet_input


class ICNetSingleImageApplier(SingleImageInferenceBase):

    @staticmethod
    def get_default_config():
        return {
            GPU_DEVICE: 0
        }

    def _load_train_config(self):
        super()._load_train_config()
        self._determine_model_input_size()

    def _validate_model_config(self, config):
        JsonConfigValidator().validate_inference_cfg(config)

    def _construct_and_fill_model(self):
        super()._construct_and_fill_model()

        self.device_ids = sly.env.remap_gpu_devices([self._config[GPU_DEVICE]])
        n_cls = (max(self.out_class_mapping.keys()) + 1)
        use_batchnorm = self.train_config[SETTINGS]['use_batchnorm']

        model = ICNet(n_classes=n_cls, input_size=self.input_size, is_batchnorm=use_batchnorm)
        WeightsRW(TaskPaths.MODEL_DIR).load_strictly(model)
        logger.info('Weights are loaded.')
        self.model = DataParallel(model, device_ids=self.device_ids)
        self.model.eval()
        self.model.cuda()

    def inference(self, img, ann):
        model_input = cv2.resize(img, self.input_size[::-1])
        model_input = make_icnet_input(model_input)
        pixelwise_probas_array = pytorch_inference.infer_per_pixel_scores_single_image(self.model, model_input,
                                                                                       out_shape=img.shape[:2])
        labels = raw_to_labels.segmentation_array_to_sly_bitmaps(self.out_class_mapping,
                                                                 np.argmax(pixelwise_probas_array, axis=2))
        pixelwise_scores_labels = raw_to_labels.segmentation_scores_to_per_class_labels(self.out_class_mapping,
                                                                                        pixelwise_probas_array)
        return Annotation(ann.img_size, labels=labels, pixelwise_scores_labels=pixelwise_scores_labels)


def main():
    single_image_applier = ICNetSingleImageApplier()
    default_inference_mode_config = InfModeFullImage.make_default_config(model_result_suffix='_icnet')
    dataset_applier = BatchInferenceApplier(single_image_inference=single_image_applier,
                                            default_inference_mode_config=default_inference_mode_config,
                                            config_validator=JsonConfigValidator())
    dataset_applier.run_inference()


if __name__ == '__main__':
    if os.getenv('DEBUG_LOG_TO_FILE', None):
        sly.add_default_logging_into_file(logger, sly.TaskPaths.DEBUG_DIR)
    sly.main_wrapper('ICNET_INFERENCE', main)
