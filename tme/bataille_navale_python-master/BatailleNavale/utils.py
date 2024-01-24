import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Union, Iterable, Callable, Optional, List

def echantillone(func: Callable[..., Union[int, float, np.ndarray]], nb_iter: int, logs: bool = True, *args) -> np.ndarray:
    if logs and nb_iter >= 10:
        start = time.time()

        values_test = [func(*args) for _ in range(10)]

        print(f'This will take approximatively {(time.time() - start) / 10 * nb_iter:.0f} seconds')
        
        values = np.array([func(*args) for _ in range(nb_iter - 10)] + values_test, dtype='float64')

    else:
        values = np.array([func(*args) for _ in range(nb_iter)], dtype='float64')

    return values


def moyenne(func: Union[Callable[..., Union[int, float, np.ndarray]], np.ndarray], nb_iter: int, logs: bool = True, *args) -> Tuple[Union[int, float, np.ndarray], np.ndarray]:
    if isinstance(func, Callable):
        values = echantillone(func, nb_iter, logs, *args)
    else:
        values = func

    return (np.sum(values, axis=0) / nb_iter, values)


def histogramme(func: Union[Callable[..., Union[int, float, np.ndarray]], np.ndarray], nb_iter: int, logs: bool = True, *args, bins: int = 50, custom_range: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    if isinstance(func, Callable):
        values = echantillone(func, nb_iter, logs, *args)
    else:
        values = func

    n, bins, _ = plt.hist(values, bins=bins, range=custom_range)
    plt.show()
    
    return (values, (n, bins))


def compare(arrays: List[np.ndarray], arrays_names: Optional[List[str]] = None):
    if arrays_names == None:
        for ind, array in enumerate(arrays):
            print(f'Array_{ind:02d} : length : {len(array)}, mean : {np.mean(array)}, median : {np.median(array)}, ecart_type : {np.nanstd(array)}')
    else:
        for name, array in zip(arrays_names, arrays):
            print(f'{name:23s} : length : {len(array)}, mean : {np.mean(array)}, median : {np.median(array)}, ecart_type : {np.nanstd(array)}')