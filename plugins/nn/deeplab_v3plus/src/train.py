# coding: utf-8

import os
import os.path

import cv2
import tensorflow as tf

import supervisely_lib as sly
import supervisely_lib.nn.dataset
from supervisely_lib.nn.config import JsonConfigValidator
from supervisely_lib.nn.hosted.trainer import SuperviselyModelTrainer
from supervisely_lib.nn.hosted.class_indexing import CONTINUE_TRAINING, TRANSFER_LEARNING
from supervisely_lib.nn.training.eval_planner import EvalPlanner
from supervisely_lib.task.progress import epoch_float

import config as config_lib
import deeplab.model as model
from deeplab.common import ModelOptions
from dataloader import DataLoader
from dataset import DeepLabV3Dataset


slim = tf.contrib.slim


class DeepLabTrainer(SuperviselyModelTrainer):
    @staticmethod
    def get_default_config():
        return {
            'dataset_tags': {
                'train': 'train',
                'val': 'val',
            },
            'batch_size': {
                'train': 1,
                'val': 1,
            },
            'data_workers': {
                'train': 1,
                'val': 1,
            },
            'input_size': {
                'width': 513,
                'height': 513,
            },
            'allow_corrupted_samples': {
                'train': 0,
                'val': 0,
            },
            'special_classes': {
                'background': 'bg',
                'neutral': 'neutral',
            },
            'epochs': 2,
            'val_every': 1,
            'lr': 0.0001,
            'weights_init_type': TRANSFER_LEARNING,  # CONTINUE_TRAINING,
            'gpu_devices': [0],
            'weight_decay': 0.00004,
            'atrous_rates': [8, 12, 18],
            'output_stride': 16,
            'backbone': 'xception_65'  # 'mobilenet_v2'
        }

    def __init__(self):
        self.bkg_input_idx = 0
        super().__init__(default_config=DeepLabTrainer.get_default_config())

    @property
    def class_title_to_idx_key(self):
        return config_lib.class_to_idx_config_key()

    @property
    def train_classes_key(self):
        return config_lib.train_classes_key()

    def _validate_train_cfg(self, config):
        JsonConfigValidator().validate_train_cfg(config)

    def _determine_model_classes(self):
        super()._determine_model_classes_segmentation(bkg_input_idx=self.bkg_input_idx)

        self.model_out_dims = max(self.class_title_to_idx.values()) + 1
        self.class_title_to_idx_with_internal_classes = self.class_title_to_idx.copy()
        self.neutral_idx = max(self.class_title_to_idx_with_internal_classes.values()) + 1
        neutral_title = self.config['special_classes'].get('neutral', None)
        if neutral_title is not None:
            self.class_title_to_idx_with_internal_classes[neutral_title] = self.neutral_idx

    def _construct_and_fill_model(self):
        # TODO: Factor out progress in base class
        progress_dummy = sly.Progress('Building model:', 1)
        progress_dummy.iter_done_report()

        self.device_ids = sly.env.remap_gpu_devices(self.config['gpu_devices'])
        src_size = self.config['input_size']
        self.input_size = (src_size['height'], src_size['width'])

        model_options = ModelOptions(
            outputs_to_num_classes={'semantic': self.model_out_dims},
            crop_size=self.input_size,
            atrous_rates=self.config['atrous_rates'],
            output_stride=self.config['output_stride']
        )

        self.inputs = tf.placeholder(tf.float32, [None] + list(self.input_size) + [3])
        self.labels = tf.placeholder(tf.int32, [None] + list(self.input_size) + [1])

        self.outputs_to_scales_to_logits = model.multi_scale_logits(
            images=self.inputs,
            model_options=model_options,
            image_pyramid=None,
            weight_decay=self.config['weight_decay'],
            is_training=True,
            fine_tune_batch_norm=False
        )

        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            self.outputs_to_scales_to_logits_val = model.multi_scale_logits(
                images=self.inputs,
                model_options=model_options,
                image_pyramid=None,
                weight_decay=self.config['weight_decay'],
                is_training=False,
                fine_tune_batch_norm=False
            )

    def _construct_loss(self):
        def create_loss(logits):
            logits = tf.image.resize_bilinear(
                logits,
                self.input_size,
                align_corners=True)
            scaled_labels = self.labels
            scaled_labels = tf.reshape(scaled_labels, shape=[-1])
            not_ignore_mask = tf.to_float(tf.not_equal(scaled_labels, self.neutral_idx)) * 1.0
            scaled_labels = tf.multiply(scaled_labels, tf.to_int32(not_ignore_mask))

            one_hot_labels = slim.one_hot_encoding(scaled_labels, self.model_out_dims, on_value=1.0, off_value=0.0)
            loss = tf.losses.softmax_cross_entropy(
                one_hot_labels,
                tf.reshape(logits, shape=[-1, self.model_out_dims]),
                weights=not_ignore_mask,
                scope=None)
            return loss

        self.loss = create_loss(self.outputs_to_scales_to_logits['semantic']['merged_logits'])
        self.val_loss = create_loss(self.outputs_to_scales_to_logits_val['semantic']['merged_logits'])


    def _construct_data_loaders(self):
        src_size = self.config['input_size']
        input_size = (src_size['height'], src_size['width'])

        self.datasets = {}
        for the_name, the_tag in self.name_to_tag.items():
            samples_lst = self._deprecated_samples_by_tag[the_tag]
            supervisely_lib.nn.dataset.ensure_samples_nonempty(samples_lst, the_tag, self.project.meta)
            the_ds = DeepLabV3Dataset(
                project_meta=self.project.meta,
                samples=samples_lst,
                out_size=input_size,
                class_mapping=self.class_title_to_idx_with_internal_classes,
                bkg_color=self.bkg_input_idx,
                allow_corrupted_cnt=self.config['allow_corrupted_samples'][the_name]
            )
            self.datasets[the_name] = the_ds
            sly.logger.info('Prepared dataset.', extra={
                'dataset_purpose': the_name, 'dataset_tag': the_tag, 'sample_cnt': len(samples_lst)
            })

        self.data_loaders = {}
        for name, need_shuffle, drop_last in [
            ('train', True, True),
            ('val', False, False),
        ]:
            # note that now batch_size from config determines batch for single device
            batch_sz = self.config['batch_size'][name]
            batch_sz_full = batch_sz * len(self.config['gpu_devices'])
            n_workers = self.config['data_workers'][name]

            num_samples = len(self.datasets[name])
            if num_samples < batch_sz_full:
                sly.logger.warn(
                    'Project contains fewer "{}" samples ({}) than the batch size ({}).'.format(name, num_samples,
                                                                                                batch_sz_full))

            self.data_loaders[name] = DataLoader(
                dataset=self.datasets[name],
                batch_size=batch_sz_full,  # it looks like multi-gpu validation works
                num_workers=n_workers,
                shuffle=need_shuffle,
                drop_last=drop_last
            )
        sly.logger.info('DataLoaders are constructed.')

        self.train_iters = len(self.data_loaders['train'])
        self.val_iters = len(self.data_loaders['val'])
        self.epochs = self.config['epochs']

    def _dump_model_weights(self, out_dir):
        model_fpath = os.path.join(out_dir, 'model_weights', 'model.ckpt')
        self.saver.save(self.session, model_fpath)

    def _validation(self, session):
        overall_val_loss = 0
        val_iters = len(self.data_loaders['val'])
        for val_it, (batch_inputs, batch_targets) in enumerate(self.data_loaders['val']):
            feed = {
                self.inputs: batch_inputs,
                self.labels: batch_targets
            }
            val_loss = session.run(self.val_loss, feed)

            overall_val_loss += val_loss
            sly.logger.info("Validation in progress", extra={'epoch': self.epoch_flt,
                                                             'val_iter': val_it,
                                                             'val_iters': val_iters})
        metrics_values_val = {
            'loss': overall_val_loss / val_iters,
        }
        sly.report_metrics_validation(self.epoch_flt, metrics_values_val)
        sly.logger.info("Validation has been finished", extra={'epoch': self.epoch_flt})
        return metrics_values_val

    def train(self):
        self.device_ids = sly.env.remap_gpu_devices(self.config['gpu_devices'])

        epochs = self.config['epochs']
        session_config = tf.ConfigProto(allow_soft_placement=True)
        # with tf.Graph().as_default(), tf.Session(config=session_config) as session:
        #     with tf.device('/gpu:0'):
        self.session = tf.Session(config=session_config)
        opt = tf.train.AdamOptimizer(self.config['lr'])
        train_op = opt.minimize(self.loss)
        init_op = tf.global_variables_initializer()
        self.session.run(init_op)

        if sly.fs.dir_empty(sly.TaskPaths.MODEL_DIR): # @TODO: implement transfer learning
            sly.logger.info('Weights were inited randomly.')
        elif self.config['weights_init_type'] == TRANSFER_LEARNING:
            vars_to_restore = slim.get_variables_to_restore()
            variables_to_restore = [v for v in vars_to_restore if
                                    ('Adam' not in v.name and '_power' not in v.name and 'logits' not in v.name)]
            re_saver = tf.train.Saver(variables_to_restore, max_to_keep=0)
            re_saver.restore(self.session, os.path.join(sly.TaskPaths.MODEL_DIR, 'model_weights', 'model.ckpt'))
        elif self.config['weights_init_type'] == CONTINUE_TRAINING:
            re_saver = tf.train.Saver(max_to_keep=0)
            re_saver.restore(self.session, os.path.join(sly.TaskPaths.MODEL_DIR, 'model_weights', 'model.ckpt'))
            sly.logger.info('Restored model weights from training')

        eval_planner = EvalPlanner(epochs, self.config['val_every'])
        progress = sly.Progress('Model training: ', epochs * self.train_iters)
        best_val_loss = float('inf')
        self.saver = tf.train.Saver(max_to_keep=0)

        for epoch in range(epochs):
            sly.logger.info("Before new epoch", extra={'epoch': self.epoch_flt})
            for train_it, (batch_inputs, batch_targets) in enumerate(self.data_loaders['train']):
                feed = {
                    self.inputs: batch_inputs,
                    self.labels: batch_targets
                }

                tl, _ = self.session.run([self.loss, train_op], feed)

                metrics_values_train = {
                    'loss': tl,
                }

                progress.iter_done_report()
                self.epoch_flt = epoch_float(epoch, train_it + 1, len(self.data_loaders['train']))
                sly.report_metrics_training(self.epoch_flt, metrics_values_train)

                if eval_planner.need_validation(self.epoch_flt):
                    sly.logger.info("Before validation", extra={'epoch': self.epoch_flt})

                    val_metrics_values = self._validation(self.session)
                    eval_planner.validation_performed()

                    val_loss = val_metrics_values['loss']

                    model_is_best = val_loss < best_val_loss
                    if model_is_best:
                        best_val_loss = val_loss
                        sly.logger.info('It\'s been determined that current model is the best one for a while.')

                    self._save_model_snapshot(model_is_best,
                                              opt_data={
                                         'epoch': self.epoch_flt,
                                         'val_metrics': val_metrics_values,
                                     })


def main():
    cv2.setNumThreads(0)
    x = DeepLabTrainer()
    x.train()


if __name__ == '__main__':
    if os.getenv('DEBUG_LOG_TO_FILE', None):
        sly.add_default_logging_into_file(sly.logger, sly.TaskPaths.DEBUG_DIR)
    sly.main_wrapper('DEEPLAB_TRAIN', main)
