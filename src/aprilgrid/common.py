from functools import wraps
from time import perf_counter
import numpy as np


def random_color() -> list:
    return np.random.randint(0, 255, 3, np.uint8).tolist()


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


def max_pool(arr, block_size: int, _max=True):

    h, w = arr.shape  # pretend we only have this
    hs, r0 = divmod(h, block_size)
    ws, r1 = divmod(w, block_size)
    pooled = arr[:h-r0, :w-r1].reshape(hs, block_size,
                                       ws, block_size)
    if _max:
        return pooled.max((1, 3))
    else:
        return pooled.min((1, 3))
