import torch
import numpy as np

from learning.logger.base import BaseLogger


class MotionTransitionLogger(BaseLogger):
    """
    See hierarchy in BaseLogger

    logger_specific_group:
        root(None):
            motion_id: int (N, L) - N: number of environment, L: length of rollout (resizeable)
        transition:
            {env_idx}: int (D) - D: number of transition (resizeable)
    """

    def __init__(self, filename: str, exp_name: str, num_env: int):
        super().__init__(filename, exp_name)

        self._ds_motion = self._base_group.create_dataset("motion_id", shape=(num_env, 0), maxshape=(num_env, None),
                                                          dtype='i4', chunks=True)
        self._trans_group = self._base_group.create_group("transition")
        for i in range(num_env):
            self._trans_group.create_dataset(str(i), shape=(0,), maxshape=(None,), dtype='i4', chunks=True)

    def update_z(self, update_env: torch.Tensor):
        cur_rollout_len = self._ds_motion.shape[1]
        for i in range(update_env.shape[0]):
            trans_ds = self._trans_group[str(update_env[i].item())]
            self._append_ds(trans_ds, np.array([cur_rollout_len]))

    def log(self, motion_indices: torch.Tensor):
        self._append_ds(self._ds_motion, motion_indices.unsqueeze(1).to("cpu").numpy(), axis=1)
