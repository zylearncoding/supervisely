# coding: utf-8
import torch
import torch.nn.functional as F

from supervisely_lib.nn.pytorch.metrics import MultiClassAccuracy


def _maybe_resize_predictions(predictions_tensor, target_tensor):
    _, _, h, w = predictions_tensor.size()
    _, ht, wt = target_tensor.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample input
        return F.interpolate(predictions_tensor, size=(ht, wt), mode="bilinear", align_corners=True)
    else:
        return predictions_tensor


def cross_entropy2d(input_batch, target, weight=None, size_average=True, ignore_index=255):
    input_batch = _maybe_resize_predictions(input_batch, target)
    _, c, _, _ = input_batch.size()

    input_batch = input_batch.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input_batch, target, weight=weight, size_average=size_average, ignore_index=ignore_index
    )
    return loss


def multi_scale_cross_entropy2d(input_batch, target, weight=None, size_average=True, scale_weight=None,
                                ignore_index=255):
    if not isinstance(input_batch, tuple):
        return cross_entropy2d(input_batch=input_batch, target=target, weight=weight,
                               size_average=size_average, ignore_index=ignore_index)

    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight is None:  # scale_weight: torch tensor type
        n_inp = len(input_batch)
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp).float()).cuda()

    loss = 0.0
    for i, inp in enumerate(input_batch):
        loss = loss + scale_weight[i] * cross_entropy2d(
            input_batch=inp, target=target, weight=weight, size_average=size_average, ignore_index=255
        )
    return loss


class MultiScaleCE:
    def __init__(self, weight=None, size_average=True, scale_weight=None, ignore_index=255):
        self.weight = weight
        self.size_averge = size_average
        self.scale_weight = scale_weight
        self.ignore_index = ignore_index

    def __call__(self, outputs, targets):
        return multi_scale_cross_entropy2d(outputs,
                                           targets,
                                           weight=self.weight,
                                           size_average=self.size_averge,
                                           scale_weight=self.scale_weight,
                                           ignore_index=self.ignore_index)


class MultiClassAccuracyUnwrapMultiscale(MultiClassAccuracy):
    """Multiclass accuracy class that accepts multiscale predictions in a tuple in addition to plain single scale."""

    def __init__(self, ignore_index=None, squeeze_targets=False, scale_idx_to_use=0):
        super().__init__(ignore_index=ignore_index, squeeze_targets=squeeze_targets)
        self._scale_idx_to_use = scale_idx_to_use

    def __call__(self, outputs, targets):
        output_requested_scale = outputs[self._scale_idx_to_use] if isinstance(outputs, tuple) else outputs
        output_requested_scale = _maybe_resize_predictions(output_requested_scale, targets)
        return super().__call__(output_requested_scale, targets)
