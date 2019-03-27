# coding: utf-8

import traceback
import string
import random
import math

from ..sly_logger import logger


def function_wrapper(f, *args, **kwargs):
    try:
        return f(*args, **kwargs)
    except Exception as e:
        logger.error(traceback.format_exc(), exc_info=True, extra={'exc_str': str(e)})
        raise e


def function_wrapper_nofail(f, *args, **kwargs):
    try:
        return f(*args, **kwargs)
    except Exception as e:
        logger.error(traceback.format_exc(), exc_info=True, extra={'exc_str': str(e)})


def generate_random_string(length):
    chars = string.ascii_letters + string.digits  # [A-z][0-9]
    return ''.join((random.choice(chars)) for _ in range(length))


def batched(seq, batch_size):
    for i in range(0, len(seq), batch_size):
        yield seq[i:i + batch_size]


def catch_silently(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


class ChunkSplitter:
    def __init__(self, tot_size, chunk_size):
        self.tot_size = tot_size
        self.chunk_size = chunk_size

    def __next__(self):
        for curr_pos in range(0, self.tot_size, self.chunk_size):
            curr_chunk_size = min(self.tot_size - curr_pos, self.chunk_size)
            yield (curr_pos, curr_chunk_size)

    def __iter__(self):
        return next(self)

    @property
    def chunk_cnt(self):
        res = int(math.ceil(self.tot_size / self.chunk_size))
        return res
