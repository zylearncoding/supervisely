# coding: utf-8
import cv2

import numpy as np
import torch
import torch.nn.functional as functional
from legacy_supervisely_lib.utils.pytorch.stuff import cuda_variable


def infer_per_pixel_softmax_single_image(model, raw_input, out_shape):
    model_input = torch.stack([raw_input], 0)  # add dim #0 (batch size 1)
    model_input = cuda_variable(model_input, volatile=True)

    output = model(model_input)
    output = functional.softmax(output, dim=1)
    output = output.data.cpu().numpy()[0]  # from batch to 3d

    pred = np.transpose(output, (1, 2, 0))
    h, w = out_shape
    return cv2.resize(pred, (w, h), cv2.INTER_LINEAR)
