import torch

from learning.logger.base import BaseLogger


class ActionLogger(BaseLogger):
    """
    See hierarchy in BaseLogger

    logger_specific_group:
        root(None):
            action: float (N, D) - N: number of collected actions (resizeable), D: dimension of action
    """

    def __init__(self, filename: str, exp_name: str, action_size: int):
        super().__init__(filename, exp_name)

        self._action_size = action_size
        if "action" not in self._base_group:
            self._base_group.create_dataset("action", shape=(0, action_size), maxshape=(None, action_size),
                                            dtype='f4', chunks=True)
        self._ds = self._base_group["action"]

    def log(self, data: torch.Tensor):
        assert isinstance(data, torch.Tensor) and data.shape[1] == self._action_size, \
            f"[{self.__class__}] data is not tensor or data.shape({data.shape}) != action_size({self._action_size})"

        self._append_ds(self._ds, data.to("cpu").numpy())
