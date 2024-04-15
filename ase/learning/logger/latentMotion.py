import torch
import numpy as np

from learning.logger.base import BaseLogger


class LatentMotionLogger(BaseLogger):
    """
    See hierarchy in BaseLogger

    logger_specific_group:
        root(None):
            latent: float (N, D) - N: number of collected latent (resizeable), D: dimension of latent
        motion_id:
            {latent_idx}: int (L) - L: number of motion id (resizeable)
    """

    def __init__(self, filename: str, exp_name: str, num_envs: int, latent_dim: int):
        super().__init__(filename, exp_name)

        self._latent_idx = np.empty(num_envs, dtype=int)
        self._ds_motion_id = None

        if "latent" not in self._base_group:
            self._base_group.create_dataset("latent", shape=(0, latent_dim), maxshape=(None, latent_dim),
                                            dtype='f4', chunks=True)
            self._base_group.create_group("motion_id")

        self._ds_latent = self._base_group["latent"]
        self._motion_id_group = self._base_group["motion_id"]

    def update_z(self, update_env: torch.Tensor, z: torch.Tensor):
        z = z.to("cpu").numpy()
        for i in range(z.shape[0]):
            env_idx = update_env[i]
            for j in range(self._ds_latent.shape[0]):
                if np.allclose(z[i], self._ds_latent[j]):
                    self._latent_idx[env_idx] = j
                    return

            # There is no same latent vector in the dataset
            self._latent_idx[env_idx] = self._ds_latent.shape[0]
            self._append_ds(self._ds_latent, z[i:i+1])
            self._motion_id_group.create_dataset(str(self._latent_idx[env_idx]), shape=(0,), maxshape=(None,), dtype='i4',
                                                 chunks=True)

    def log(self, motion_indices: torch.Tensor):
        assert motion_indices.shape[0] == self._latent_idx.shape[0], \
            f"latent_idx({self._latent_idx.shape[0]}) != data({motion_indices.shape[0]})"

        for i in range(motion_indices.shape[0]):
            motion_ids = self._motion_id_group[str(self._latent_idx[i])]
            self._append_ds(motion_ids, motion_indices[i:i + 1].to("cpu").numpy())
