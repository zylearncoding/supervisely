# coding: utf-8
import os.path

import torch

from legacy_supervisely_lib.utils.pytorch.stuff import upgraded_load_state_dict


class WeightsRW:
    def __init__(self, model_dir):
        self._model_dir = model_dir

    @property
    def _weights_fpath(self):
        return os.path.join(self._model_dir, 'model.pt')

    def save(self, model):
        torch.save(model.state_dict(), self._weights_fpath)

    def load_for_transfer_learning(self, model, delete_matching_substrings=None, logger=None):
        delete_matching_substrings = delete_matching_substrings or []
        loaded_model = torch.load(self._weights_fpath)

        # Remove the layers matching the requested patterns (usually done for the head layers in transfer learning).
        # Make an explicit set for easier logging.
        to_delete = set(el for el in loaded_model.keys() if
                        any(delete_substring in el for delete_substring in delete_matching_substrings))
        loaded_model = {k: v for k, v in loaded_model.items() if k not in to_delete}
        if len(to_delete) > 0 and logger is not None:
            logger.info('Skip weight init for output layers.', extra={'layer_names': sorted(to_delete)})

        upgraded_load_state_dict(model, loaded_model, strict=False)
        return model

    def load_strictly(self, model):
        loaded_model = torch.load(self._weights_fpath)
        upgraded_load_state_dict(model, loaded_model, strict=True)
        return model
