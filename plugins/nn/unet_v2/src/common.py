# coding: utf-8
from torch.nn import DataParallel
from supervisely_lib import logger
from supervisely_lib.nn.pytorch.weights import WeightsRW
from supervisely_lib.nn import config as sly_nn_config
from unet import construct_unet


class UnetJsonConfigValidator(sly_nn_config.JsonConfigValidator):
    def validate_train_cfg(self, config):
        super().validate_train_cfg(config)
        sp_classes = config['special_classes']
        if len(set(sp_classes.values())) != len(sp_classes):
            raise RuntimeError('Non-unique special classes in train config.')


def create_model(n_cls, device_ids):
    logger.info('Will construct model.')
    model = construct_unet(n_cls=n_cls)
    logger.info('Model has been constructed (w/out weights).')
    model = DataParallel(model, device_ids=device_ids).cuda()
    logger.info('Model has been loaded into GPU(s).', extra={'remapped_device_ids': device_ids})
    return model


def create_model_for_inference(n_cls, device_ids, model_dir):
    model = create_model(n_cls, device_ids)
    model = WeightsRW(model_dir).load_strictly(model)
    model.eval()
    return model
