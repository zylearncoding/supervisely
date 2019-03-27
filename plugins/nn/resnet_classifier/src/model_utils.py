# coding: utf-8
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

from supervisely_lib.sly_logger import logger


num_layers_to_model = {
    18: resnet18,
    34: resnet34,
    50: resnet50,
    101: resnet101,
    152: resnet152
}


def create_model(num_layers, n_cls, device_ids):
    logger.info('Will construct ResNet{} model.'.format(num_layers))
    model = num_layers_to_model[num_layers]()
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, n_cls)

    logger.info('Model has been constructed (w/out weights).')
    model = DataParallel(model, device_ids=device_ids).cuda()
    logger.info('Model has been loaded into GPU(s).', extra={'remapped_device_ids': device_ids})
    return model