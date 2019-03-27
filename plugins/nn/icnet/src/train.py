# coding: utf-8

import os
from collections import defaultdict

import cv2
import torch

from torch.optim import Adam
from torch.nn import DataParallel
from torch.utils.data import DataLoader

import supervisely_lib as sly
from supervisely_lib import logger
from supervisely_lib.nn.config import JsonConfigValidator
from supervisely_lib.nn.hosted.trainer import SuperviselyModelTrainer
from supervisely_lib.nn.training.eval_planner import EvalPlanner
from supervisely_lib.nn.dataset import SlyDataset, ensure_samples_nonempty
from supervisely_lib.nn.hosted.class_indexing import CONTINUE_TRAINING, TRANSFER_LEARNING
from supervisely_lib.nn.pytorch.cuda import cuda_variable
from supervisely_lib.nn.pytorch.weights import WeightsRW
from supervisely_lib.task.progress import epoch_float

from icnet import ICNet, make_icnet_input
from metrics import MultiScaleCE, MultiClassAccuracyUnwrapMultiscale


class ICNetDataset(SlyDataset):
    def _get_sample_impl(self, img_fpath, ann_fpath):
        img, gt = super()._get_sample_impl(img_fpath=img_fpath, ann_fpath=ann_fpath)
        return make_icnet_input(img), gt


class LRPolicyWithPatience:
    def __init__(self, optim_cls, init_lr, patience, lr_divisor, model):
        self.optimizer = None
        self.optim_cls = optim_cls
        self.lr = init_lr
        self.patience = patience
        self.lr_divisor = lr_divisor
        self.losses = []
        self.last_reset_idx = 0

        logger.info('Selected optimizer.', extra={'optim_class': self.optim_cls.__name__})
        self._reset(model)

    def _reset(self, model):
        self.optimizer = self.optim_cls(model.parameters(), lr=self.lr)
        logger.info('Learning Rate has been updated.', extra={'lr': self.lr})

    def reset_if_needed(self, new_loss, model):
        self.losses.append(new_loss)
        no_recent_update = (len(self.losses) - self.last_reset_idx) > self.patience
        no_loss_improvement = min(self.losses[-self.patience:]) > min(self.losses)
        if no_recent_update and no_loss_improvement:
            self.lr /= float(self.lr_divisor)
            self._reset(model)
            self.last_reset_idx = len(self.losses)


class ICNetTrainer(SuperviselyModelTrainer):
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
                'width': 2049,
                'height': 1025,
            },
            'allow_corrupted_samples': {
                'train': 0,
                'val': 0,
            },
            'special_classes': {
                'neutral': 'neutral'
            },
            'epochs': 2,
            'val_every': 1,
            'lr': 0.0001,
            'lr_decreasing': {
                'patience': 1000,
                'lr_divisor': 5,
            },
            'weights_init_type': TRANSFER_LEARNING,  # CONTINUE_TRAINING,
            'gpu_devices': [0],
            'use_batchnorm': True
        }

    def __init__(self):
        self.bkg_input_idx = 0
        super().__init__(default_config=ICNetTrainer.get_default_config())

    def _validate_train_cfg(self, config):
        JsonConfigValidator().validate_train_cfg(config)

    def _determine_model_classes(self):
        super()._determine_model_classes_segmentation(bkg_input_idx=None)

        self.model_out_dims = max(self.class_title_to_idx.values()) + 1
        self.class_title_to_idx_with_internal_classes = self.class_title_to_idx.copy()
        self.neutral_idx = max(self.class_title_to_idx_with_internal_classes.values()) + 1
        neutral_title = self.config.get('special_classes', {}).get('neutral', None)
        if neutral_title is not None:
            self.class_title_to_idx_with_internal_classes[neutral_title] = self.neutral_idx

    def _construct_and_fill_model(self):
        progress_dummy = sly.Progress('Building model:', 1)
        progress_dummy.iter_done_report()

        self.device_ids = sly.env.remap_gpu_devices(self.config['gpu_devices'])
        src_size = self.config['input_size']
        self.input_size = (src_size['height'], src_size['width'])

        self.raw_model = ICNet(n_classes=self.model_out_dims,
                               input_size=self.input_size,
                               is_batchnorm=self.config['use_batchnorm'])

        if sly.fs.dir_empty(sly.TaskPaths.MODEL_DIR):
            sly.logger.info('Weights will not be inited.')
        else:
            wi_type = self.config['weights_init_type']
            ewit = {'weights_init_type': wi_type}
            logger.info('Weights will be inited from given model.', extra=ewit)

            weights_rw = WeightsRW(sly.TaskPaths.MODEL_DIR)
            if wi_type == TRANSFER_LEARNING:
                self.raw_model = weights_rw.load_for_transfer_learning(self.raw_model,
                                                                       ignore_matching_layers=['classif'],
                                                                       logger=logger)
            elif wi_type == CONTINUE_TRAINING:
                self.raw_model = weights_rw.load_strictly(self.raw_model)

            logger.info('Weights are loaded.', extra=ewit)
        self.raw_model.cuda()
        self.model = DataParallel(self.raw_model, device_ids=self.device_ids)

    def _construct_loss(self):
        self.metrics = {
            'accuracy': MultiClassAccuracyUnwrapMultiscale(ignore_index=self.neutral_idx)
        }

        self.criterion = MultiScaleCE(ignore_index=self.neutral_idx)

        self.val_metrics = {
            'loss': self.criterion,
            **self.metrics
        }
        logger.info('Selected metrics.', extra={'metrics': list(self.metrics.keys())})

    def _construct_data_loaders(self):
        self.device_ids = sly.env.remap_gpu_devices(self.config['gpu_devices'])

        src_size = self.config['input_size']
        input_size = (src_size['height'], src_size['width'])

        self.pytorch_datasets = {}
        for the_name, the_tag in self.name_to_tag.items():
            samples_lst = self._deprecated_samples_by_tag[the_tag]
            ensure_samples_nonempty(samples_lst, the_tag, self.project.meta)
            the_ds = ICNetDataset(
                project_meta=self.project.meta,
                samples=samples_lst,
                out_size=input_size,
                class_mapping=self.class_title_to_idx_with_internal_classes,
                bkg_color=self.bkg_input_idx,
                allow_corrupted_cnt=self.config['allow_corrupted_samples'][the_name]
            )
            self.pytorch_datasets[the_name] = the_ds
            logger.info('Prepared dataset.', extra={
                'dataset_purpose': the_name, 'dataset_tag': the_tag, 'sample_cnt': len(samples_lst)
            })

        self.data_loaders = {}
        for name, need_shuffle in [
            ('train', True),
            ('val', False),
        ]:
            # note that now batch_size from config determines batch for single device
            batch_sz = self.config['batch_size'][name]
            batch_sz_full = batch_sz * len(self.device_ids)
            n_workers = self.config['data_workers'][name]
            self.data_loaders[name] = DataLoader(
                dataset=self.pytorch_datasets[name],
                batch_size=batch_sz_full,  # it looks like multi-gpu validation works
                num_workers=n_workers,
                shuffle=need_shuffle,
            )
        logger.info('DataLoaders are constructed.')

        self.train_iters = len(self.data_loaders['train'])
        self.val_iters = len(self.data_loaders['val'])
        self.epochs = self.config['epochs']
        self.eval_planner = EvalPlanner(epochs=self.epochs, val_every=self.config['val_every'])

    def _dump_model_weights(self, out_dir):
        WeightsRW(out_dir).save(self.raw_model)

    def _validation(self):
        sly.logger.info("Before validation", extra={'epoch': self.epoch_flt})

        self.model.eval()

        metrics_values = defaultdict(int)
        samples_cnt = 0

        for val_it, (inputs, targets) in enumerate(self.data_loaders['val']):
            inputs, targets = cuda_variable(inputs, volatile=True), cuda_variable(targets)
            outputs = self.model(inputs)
            full_batch_size = inputs.size(0)
            for name, metric in self.val_metrics.items():
                metric_value = metric(outputs, targets)
                if isinstance(metric_value, torch.Tensor):
                    metric_value = metric_value.item()
                metrics_values[name] += metric_value * full_batch_size
            samples_cnt += full_batch_size

            sly.logger.info("Validation in progress", extra={'epoch': self.epoch_flt,
                                                             'val_iter': val_it, 'val_iters': self.val_iters})

        for name in metrics_values:
            metrics_values[name] /= float(samples_cnt)

        sly.report_metrics_validation(self.epoch_flt, metrics_values)

        self.model.train()
        sly.logger.info("Validation has been finished", extra={'epoch': self.epoch_flt})
        return metrics_values

    def train(self):
        progress = sly.Progress('Model training: ', self.epochs * self.train_iters)
        self.model.train()

        lr_decr = self.config['lr_decreasing']
        policy = LRPolicyWithPatience(
            optim_cls=Adam,
            init_lr=self.config['lr'],
            patience=lr_decr['patience'],
            lr_divisor=lr_decr['lr_divisor'],
            model=self.model
        )
        best_val_loss = float('inf')

        for epoch in range(self.epochs):
            sly.logger.info("Before new epoch", extra={'epoch': self.epoch_flt})

            for train_it, (inputs_cpu, targets_cpu) in enumerate(self.data_loaders['train']):
                inputs, targets = cuda_variable(inputs_cpu), cuda_variable(targets_cpu)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                policy.optimizer.zero_grad()
                loss.backward()
                policy.optimizer.step()

                metric_values_train = {'loss': loss.item()}
                for name, metric in self.metrics.items():
                    metric_values_train[name] = metric(outputs, targets)

                progress.iter_done_report()

                self.epoch_flt = epoch_float(epoch, train_it + 1, self.train_iters)
                sly.report_metrics_training(self.epoch_flt, metric_values_train)

                if self.eval_planner.need_validation(self.epoch_flt):
                    metrics_values_val = self._validation()
                    self.eval_planner.validation_performed()

                    val_loss = metrics_values_val['loss']
                    model_is_best = val_loss < best_val_loss
                    if model_is_best:
                        best_val_loss = val_loss
                        sly.logger.info('It\'s been determined that current model is the best one for a while.')

                    self._save_model_snapshot(model_is_best, opt_data={
                        'epoch': self.epoch_flt,
                        'val_metrics': metrics_values_val,
                    })

                    policy.reset_if_needed(val_loss, self.model)

            logger.info("Epoch was finished", extra={'epoch': self.epoch_flt})


def main():
    cv2.setNumThreads(0)
    x = ICNetTrainer()
    x.train()


if __name__ == '__main__':
    if os.getenv('DEBUG_LOG_TO_FILE', None):
        sly.add_default_logging_into_file(sly.logger, sly.TaskPaths.DEBUG_DIR)
    sly.main_wrapper('ICNET_TRAIN', main)
