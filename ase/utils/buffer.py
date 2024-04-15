from typing import List

import torch

from poselib.poselib.motion_lib import MotionLib


class SingleTensorBuffer:
    """A buffer that
        1. stores data in to a single tensor.
        2. supports random sampling.
    Indexing is done by slicing the first dimension of tensor.
    """
    def __init__(self, size, device):
        self._top = 0
        self._count = 0
        self._size = size
        self._device = device
        self._buf = None

    def reset(self):
        self._top = 0
        self._count = 0

    @property
    def size(self):
        return self._size

    @property
    def count(self):
        return self._count

    def store(self, data: torch.Tensor):
        assert data.shape[0] <= self._size, \
            f"[{self.__class__}] data.shape[0]({data.shape[0]}) > self._size({self._size})"
        if self._buf is None:
            self._buf = torch.zeros((self._size, *data.shape[1:]), dtype=data.dtype, device=self._device)
        assert data.shape[1:] == self._buf.shape[1:], f"[{self.__class__}] data.shape[1:] != self._buf.shape[1:]"

        next_top = min(self._top + data.shape[0], self._size)
        if not self._top == next_top:
            self._buf[self._top:next_top].copy_(data[:next_top - self._top])
        remainder = data.shape[0] - (next_top - self._top)
        if remainder > 0:
            self._buf[:remainder].copy_(data[next_top - self._top:])
            next_top = remainder

        self._top = next_top
        self._count = min(self._size, self._count + data.shape[0])

    def sample(self, n):
        idx = torch.randint(0, self._count, (n,))
        return self._buf[idx].clone()


class MotionLibFetcher:
    def __init__(self, traj_len: int, dt: float, device: torch.device, motion_file: str, dof_body_ids: List[float],
                 dof_offsets: List[float], key_body_ids: List[int]):
        self._traj_len = traj_len
        self._dt = dt
        self._device = device

        config = {
            'motion_file': motion_file,
            'dof_body_ids': dof_body_ids,
            'dof_offsets': dof_offsets,
            'key_body_ids': key_body_ids,
            'device': device
        }
        self._motion_lib = MotionLib(**config)

    def fetch_traj(self, n: int = 1):
        motion_ids = self._motion_lib.sample_motions(n)

        motion_length = self._dt * (self._traj_len + 1)
        start_time = self._motion_lib.sample_time(motion_ids, truncate_time=motion_length)
        end_time = (start_time + motion_length).unsqueeze(-1)
        time_steps = - self._dt * torch.arange(0, self._traj_len, device=self._device)
        capture_time = (time_steps + end_time).view(-1)

        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._traj_len]).view(-1)
        # [traj1, traj1, ..., traj2, traj2, ...]
        return self._motion_lib.get_motion_state(motion_ids, capture_time)

    def fetch_snapshot(self, n: int = 1):
        motion_ids = self._motion_lib.sample_motions(n)
        motion_times = self._motion_lib.sample_time(motion_ids)
        return self._motion_lib.get_motion_state(motion_ids, motion_times)

    @property
    def motion_lib(self):
        return self._motion_lib

    @staticmethod
    def demo_fetcher_config(cls, algo_conf):
        return {
            # Demo dimension
            'traj_len': cls._disc_obs_traj_len,
            'dt': cls.vec_env.dt,

            # Motion Lib
            'motion_file': algo_conf['motion_file'],
            'dof_body_ids': algo_conf['joint_information']['dof_body_ids'],
            'dof_offsets': cls._dof_offsets,
            'key_body_ids': cls._key_body_ids,
            'device': cls.device
        }


@torch.jit.script
class TensorFIFO:
    def __init__(self, max_size: int):
        self._q = []
        self._max_size = max_size

    def push(self, item: torch.Tensor):
        self._q.insert(0, item)
        if len(self._q) > self._max_size:
            self._q.pop()

    def set_row(self, idx: int, item: torch.Tensor, set_flag: torch.Tensor):
        """Set certain row of the tensor in the FIFO.
        If not, do anything.

        :param idx: index of the item to be set.
        :param item: tensor to be set.
        :param set_flag:
        :return:
        """
        assert idx < len(self._q), f"[TensorFIFO] index out of range: {idx} >= {len(self._q)}"
        assert item.shape[0] == set_flag.shape[0], \
            f"[TensorFIFO] set_flag shape mismatch: {item.shape[0]} != {set_flag.shape[0]}"
        assert item.shape == self._q[0].shape, \
            f"[TensorFIFO] item shape mismatch: {item.shape} != {self._q[0].shape}"

        if len(set_flag.shape) == 1:
            set_flag = set_flag.unsqueeze(1)

        self._q[idx] = torch.where(set_flag, item, self._q[idx])

    @property
    def max_len(self):
        return self._max_size

    @property
    def list(self):
        return self._q

    def __getitem__(self, idx):
        return self._q[idx]

    def __len__(self):
        return len(self._q)


@torch.jit.script
class TensorHistoryFIFO:
    def __init__(self, max_size: int):
        self._max_size = max_size
        self._q = TensorFIFO(max_size)

    def push_on_reset(self, x: torch.Tensor, resets: torch.Tensor):
        assert x.shape[0] == resets.shape[0], f"[TensorHistoryFIFO] shape mismatch: {x.shape[0]} != {resets.shape[0]}"
        if len(self._q) == 0:
            for i in range(self._q.max_len):
                self._q.push(x)
        elif torch.any(resets):
            for i in range(self._q.max_len):
                self._q.set_row(i, x, resets)

    def push(self, x: torch.Tensor):
        if len(self._q) == 0:
            for i in range(self._q.max_len):
                self._q.push(x)
        else:
            if len(self._q) != self._max_size:
                raise ValueError(f"[TensorHistoryFIFO] FIFO is empty. Cannot push.")
            self._q.push(x)

    @property
    def history(self):
        return torch.cat(self._q.list, dim=1)

    def __len__(self):
        return len(self._q)
