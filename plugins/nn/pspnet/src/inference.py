# coding: utf-8

import os
import cv2
import numpy as np
import tensorflow as tf

import supervisely_lib as sly

from supervisely_lib.annotation.annotation import Annotation
from supervisely_lib.sly_logger import logger
from supervisely_lib.nn.hosted.constants import SETTINGS
from supervisely_lib.nn.hosted.inference_single_image import SingleImageInferenceBase, GPU_DEVICE
from supervisely_lib.nn.hosted.inference_batch import BatchInferenceApplier
from supervisely_lib.nn.hosted.inference_modes import InfModeFullImage
from supervisely_lib.nn import raw_to_labels
from supervisely_lib.nn.config import JsonConfigValidator

from model import PSPNet101, PSPNet50
from tools import preprocess, decode_to_n_channels


MODEL_ARCH = 'model_arch'
USE_FLIP = 'use_flip'

ModelArchitectures = {
    'pspnet50': PSPNet50,
    'pspnet101': PSPNet101
}


class PSPNetSingleImageApplier(SingleImageInferenceBase):

    @staticmethod
    def get_default_config():
        return {
            GPU_DEVICE: 0,
            USE_FLIP: True
        }

    def _load_train_config(self):
        super()._load_train_config()
        self._determine_model_input_size()
        model_arch_name = self.train_config[SETTINGS][MODEL_ARCH]
        self.model_class = ModelArchitectures.get(model_arch_name)
        if self.model_class is None:
            raise RuntimeError(f'Unknown PSPNet architecture version: {model_arch_name!r}. Supported architectures: '
                               f'{sorted(ModelArchitectures.keys())!r}.')
        self.model_checkpoint_path = os.path.join(sly.TaskPaths.MODEL_DIR)

    def _validate_model_config(self, config):
        JsonConfigValidator().validate_inference_cfg(config)

    def _construct_and_fill_model(self):
        super()._construct_and_fill_model()

        sly.env.remap_gpu_devices([self._config[GPU_DEVICE]])
        n_cls = (max(self.out_class_mapping.keys()) + 1)

        self.image_tensor = tf.placeholder("float32", list(self.input_size) + [3])
        img = preprocess(self.image_tensor, self.input_size[0], self.input_size[1])

        net = self.model_class({'data': img}, is_training=False, num_classes=n_cls)

        # Predictions.
        raw_output = net.layers['conv6']

        if self._config[USE_FLIP]:
            with tf.variable_scope('', reuse=True):
                flipped_img = tf.image.flip_left_right(tf.squeeze(self.image_tensor))
                flipped_img = tf.expand_dims(flipped_img, dim=0)
                net_flip = self.model_class({'data': flipped_img}, is_training=False, num_classes=n_cls)
            flipped_output = tf.image.flip_left_right(tf.squeeze(net_flip.layers['conv6']))
            flipped_output = tf.expand_dims(flipped_output, dim=0)
            raw_output = tf.add_n([raw_output, flipped_output])

        raw_output_up = tf.image.resize_bilinear(raw_output, size=self.input_size, align_corners=True)
        raw_output_up = tf.image.crop_to_bounding_box(raw_output_up, 0, 0, self.input_size[0], self.input_size[1])
        raw_output_up = tf.argmax(raw_output_up, axis=3)
        self.pred_holder = decode_to_n_channels(raw_output_up, self.input_size, n_cls)

        # Init tf Session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        init = tf.global_variables_initializer()

        self.sess.run(init)
        restore_var = tf.global_variables()
        loader = tf.train.Saver(var_list=restore_var)
        loader.restore(self.sess, os.path.join(sly.TaskPaths.MODEL_DIR, 'model.ckpt-0'))

    def inference(self, img, ann):
        # Rescale with proportions and pad image to model input size
        min_side_coef = min(self.input_size[0]/float(img.shape[0]), self.input_size[1]/float(img.shape[1]))
        img_resized = cv2.resize(img, dsize=None, fx=min_side_coef, fy=min_side_coef, interpolation=cv2.INTER_CUBIC)
        img_padded = cv2.copyMakeBorder(img_resized,
                                        0, self.input_size[0] - img_resized.shape[0],
                                        0, self.input_size[1] - img_resized.shape[1],
                                        cv2.BORDER_CONSTANT, value=0)

        preds = self.sess.run(self.pred_holder, feed_dict={self.image_tensor: img_padded})[0]
        preds = np.argmax(preds, axis=2)

        # Un-pad and rescale prediction to original image size
        preds = preds[0:img_resized.shape[0], 0:img_resized.shape[1]]
        preds = cv2.resize(preds, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        labels = raw_to_labels.segmentation_array_to_sly_bitmaps(self.out_class_mapping, preds)
        return Annotation(img_size=ann.img_size, labels=labels)


def main():
    single_image_applier = PSPNetSingleImageApplier()
    default_inference_mode_config = InfModeFullImage.make_default_config(model_result_suffix='_pspnet')
    dataset_applier = BatchInferenceApplier(single_image_inference=single_image_applier,
                                            default_inference_mode_config=default_inference_mode_config,
                                            config_validator=JsonConfigValidator())
    dataset_applier.run_inference()


if __name__ == '__main__':
    if os.getenv('DEBUG_LOG_TO_FILE', None):
        sly.add_default_logging_into_file(logger, sly.TaskPaths.DEBUG_DIR)
    sly.main_wrapper('PSPNET_INFERENCE', main)