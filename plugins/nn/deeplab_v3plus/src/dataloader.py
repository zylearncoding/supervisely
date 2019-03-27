# coding: utf-8

import numpy as np
from concurrent.futures import ThreadPoolExecutor


class RandomSampler(object):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(np.random.permutation(np.arange(len(self.data_source))))

    def __len__(self):
        return len(self.data_source)


class SequentialSampler(object):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class BatchSampler(object):
    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


def collate_fn(batch):
    imgs = []
    targets = []
    for img, target in batch:
        imgs.append(img)
        targets.append(target)
    return imgs, targets


class DataLoader(object):
    def __init__(self, dataset, batch_size=1, sampler=None, batch_sampler=None, shuffle=True,
                 num_workers=None, drop_last=True):
        self.dataset, self.batch_size, self.num_workers = dataset, batch_size, num_workers
        self.drop_last = drop_last

        if batch_sampler is None:
            if sampler is None:
                if shuffle == True:
                    sampler = RandomSampler(dataset)
                else:
                    sampler = SequentialSampler(dataset)
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.sampler = sampler
        self.batch_sampler = batch_sampler

    def __len__(self):
        return len(self.batch_sampler)

    def get_batch(self, indices):
        res = collate_fn([self.dataset[i] for i in indices])
        return res

    def __iter__(self):
        with ThreadPoolExecutor(max_workers=self.num_workers) as e:
            for batch in e.map(self.get_batch, iter(self.batch_sampler)):
                yield batch
