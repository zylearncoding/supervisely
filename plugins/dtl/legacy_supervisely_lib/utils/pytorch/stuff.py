# coding: utf-8

import torch
from torch.nn import Parameter
from torch.autograd import Variable

# no_grad was introduced in torch 0.4, so we need to check whether it is available in case we need to fall back to older
# API for creating variables without gradients. Do it once at import time and cache the result,
_TORCH_HAS_NO_GRAD = 'no_grad' in dir(torch)


def cuda_variable(x, volatile=False):
    if _TORCH_HAS_NO_GRAD:
        if volatile:
            with torch.no_grad():
                v = Variable(x)
        else:
            v = Variable(x)
    else:
        v = Variable(x, volatile=volatile)

    return v.cuda()


def upgraded_load_state_dict(model, state_dict, strict=True):
    """Copies parameters and buffers from :attr:`state_dict` into
    this module and its descendants. If :attr:`strict` is ``True`` then
    the keys of :attr:`state_dict` must exactly match the keys returned
    by this module's :func:`state_dict()` function.

    Arguments:
        model: model
        state_dict (dict): A dict containing parameters and
            persistent buffers.
        strict (bool): Strictly enforce that the keys in :attr:`state_dict`
            match the keys returned by this module's `:func:`state_dict()`
            function.
    """
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name in own_state:
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except Exception:
                raise RuntimeError('While copying the parameter named {}, '
                                   'whose dimensions in the model are {} and '
                                   'whose dimensions in the checkpoint are {}.'
                                   .format(name, own_state[name].size(), param.size()))
        elif strict:
            raise KeyError('unexpected key "{}" in state_dict'
                           .format(name))
    if strict:
        missing = set(own_state.keys()) - set(state_dict.keys())
        if len(missing) > 0:
            raise KeyError('missing keys in state_dict: "{}"'.format(missing))
