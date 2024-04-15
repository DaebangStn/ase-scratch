import numpy as np
import h5py
from h5py import Dataset


class BaseLogger:
    """
    Hierarchy:
        [filename].hdf5 --> [experiment_name]/[logger_class_name]/[logger_specific_group]/[data]
                                                    |
                                             self._base_group
    """
    def __init__(self, filename: str, experiment_name: str):
        log_file = h5py.File(filename + '.hdf5', 'a')
        print(f"===> Loaded {filename}.hdf5 for logging {experiment_name}")

        if experiment_name not in log_file:
            log_file.create_group(experiment_name)
        group = log_file[experiment_name]
        c_name = self.__class__.__name__
        if c_name not in group:
            group.create_group(c_name)

        self._base_group = group[c_name]

    def log(self, data):
        pass

    @staticmethod
    def _append_ds(ds: Dataset, data: np.ndarray, axis=0):
        new_len = data.shape[axis] + ds.shape[axis]
        new_shape = list(ds.shape)
        new_shape[axis] = new_len
        ds.resize(tuple(new_shape))

        slc = [slice(None)] * data.ndim
        slc[axis] = slice(ds.shape[axis] - data.shape[axis], None)  # [:, :, ..., -data.shape[axis]:, :, ...]

        ds[tuple(slc)] = data
