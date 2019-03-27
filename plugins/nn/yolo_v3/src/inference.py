# coding: utf-8

import os

import supervisely_lib as sly
from supervisely_lib.nn.hosted.inference_single_image import SingleImageInferenceBase, GPU_DEVICE
from supervisely_lib.nn.hosted.inference_batch import BatchInferenceApplier
from supervisely_lib.nn.hosted.inference_modes import InfModeFullImage, CONFIDENCE
from common import YoloJsonConfigValidator, infer_on_image, construct_model
import common


class YOLOSingleImageApplier(SingleImageInferenceBase):

    def __init__(self):
        sly.logger.info('YOLOv3 inference init started.')
        super().__init__()
        self.confidence_thresh = self._config['min_confidence_threshold']
        sly.logger.info('YOLOv3 inference init done.')

    @staticmethod
    def get_default_config():
        return {
            'min_confidence_threshold': 0.5,
            GPU_DEVICE: 0,
            'confidence_tag_name': CONFIDENCE
        }

    @property
    def train_classes_key(self):
        return common.train_classes_key()

    @property
    def class_title_to_idx_key(self):
        return common.class_to_idx_config_key()

    def _model_out_obj_tags(self):
        return sly.TagMetaCollection(items=[self.confidence_tag_meta])

    def _load_train_config(self):
        self.confidence_tag_meta = sly.TagMeta(self._config['confidence_tag_name'], sly.TagValueType.ANY_NUMBER)
        super()._load_train_config()

    def _validate_model_config(self, config):
        YoloJsonConfigValidator().validate_inference_cfg(config)

    def _construct_and_fill_model(self):
        super()._construct_and_fill_model()
        self.device_ids = sly.env.remap_gpu_devices([self._config[GPU_DEVICE]])
        self.model = construct_model(sly.TaskPaths.MODEL_DIR)
        sly.logger.info('Weights are loaded.')

    def inference(self, img, ann):
        labels = infer_on_image(image=img,
                                model=self.model,
                                idx_to_class=self.out_class_mapping,
                                confidence_thresh=self.confidence_thresh,
                                confidence_tag_meta=self.confidence_tag_meta,
                                num_classes=len(self.class_title_to_idx))
        return sly.Annotation(ann.img_size, labels=labels)


def main():
    single_image_applier = YOLOSingleImageApplier()
    default_inference_mode_config = InfModeFullImage.make_default_config(model_result_suffix='_yolo')
    dataset_applier = BatchInferenceApplier(single_image_inference=single_image_applier,
                                            default_inference_mode_config=default_inference_mode_config,
                                            config_validator=YoloJsonConfigValidator())
    dataset_applier.run_inference()


if __name__ == '__main__':
    if os.getenv('DEBUG_LOG_TO_FILE', None):
        sly.add_default_logging_into_file(sly.logger, sly.TaskPaths.DEBUG_DIR)
    sly.main_wrapper('YOLO_V3_INFERENCE', main)
