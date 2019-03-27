# coding: utf-8

import cv2
import json

from supervisely_lib.nn.hosted.constants import SETTINGS

from torch.nn import functional

from dataset import input_image_normalizer


def infer_on_img(img, input_size, model):
    x = cv2.resize(img, input_size[::-1])
    x = input_image_normalizer(x)
    output = model(x.unsqueeze_(0).cuda())
    output = functional.softmax(output, dim=1)

    output = output.data.cpu().numpy()[0]
    return output


def create_classes(cls_list):
    class_title_to_idx = {}
    in_project_titles = sorted(cls_list)

    for i, title in enumerate(in_project_titles):
        class_title_to_idx[title] = i

    if len(set(class_title_to_idx.values())) != len(class_title_to_idx):
        raise RuntimeError('Unable to construct internal color mapping for classes.')

    return class_title_to_idx, in_project_titles


# copypaste from tf_object_detection(num_layers <=> model_configuration)
def determine_resnet_model_configuration(model_config_fpath):
    try:
        with open(model_config_fpath) as fin:
            model_config = json.load(fin)
    except FileNotFoundError:
        raise RuntimeError('Unable to run inference, config from training was not found.')

    # The old version of this code stored num_layers inside the training config. For backwards
    # compatibility we accept both locations. If both are present, make sure they are consistent.
    # still accept that field in the input, but clear it before writing out the config in a new
    # format. Also, if num_layers is present in the training config, make sure it is consistent with the
    # values coming from the model itself.
    model_configuration_model_root = model_config.get('num_layers', None)
    model_configuration_model_config = model_config.get(SETTINGS, {}).get('num_layers', None)
    if model_configuration_model_root is None and model_configuration_model_config is None:
        raise RuntimeError('Plugin misconfigured. num_layers field is missing from internal config.json')
    elif (model_configuration_model_root is not None and
          model_configuration_model_config is not None and
          model_configuration_model_root != model_configuration_model_config):
        raise RuntimeError(
            'Plugin misconfigured. Inconsistent duplicate num_layers field in internal config.json')
    else:
        return (model_configuration_model_root if model_configuration_model_root is not None
        else model_configuration_model_config)

