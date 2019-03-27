# coding: utf-8
import math

class EvalPlanner:
    def __init__(self, epochs, val_every):
        self.epochs = epochs
        self.val_every = val_every
        self.total_val_cnt = self.validations_cnt(epochs, val_every)
        self._val_cnt = 0

    @property
    def performed_val_cnt(self):
        return self._val_cnt

    @staticmethod
    def validations_cnt(ep_float, val_every):
        res = math.floor(ep_float / val_every + 1e-9)
        return res

    def validation_performed(self):
        self._val_cnt += 1

    def need_validation(self, epoch_flt):
        req_val_cnt = self.validations_cnt(epoch_flt, self.val_every)
        need_val = req_val_cnt > self._val_cnt
        return need_val
