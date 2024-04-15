from typing import Tuple

import torch

from poselib.poselib.motion_lib import MotionLib, USE_CACHE
from poselib.poselib.visualization.plt_plotter import Matplotlib3DPlotter
from poselib.poselib.visualization.skeleton_plotter_tasks import Draw3DSkeletonMotion
from utils.buffer import TensorHistoryFIFO
from utils.angle import calc_heading_quat_inv, quat_rotate


class Matcher:
    def __init__(self, motion_lib: MotionLib, traj_len: int, sim_dt: float, device: torch.device,
                 plot_env0: bool = False):
        assert USE_CACHE, \
            f"Since MotionLib must located in the GPU, USE_CACHE of MotionLib must be True. But it is {USE_CACHE}."

        for i, dt in enumerate(motion_lib.motion_dt):
            if (sim_dt - dt) > 1e-4:
                raise ValueError(f"[Error] sim_dt({sim_dt}) is not equal to motion_dt[{i}]({dt})."
                                 f"Motion file name: {motion_lib.motion_files[i]}")

        self._device = device
        self._traj_len = traj_len

        self._stacked_motion_traj = None

        self._motion_start_idx = None

        self._dof_body_ids = None
        self._dof_offsets = None
        self._num_dof = None
        self._key_body_ids = None
        self._motions = None
        self._motions_plot = None

        self._build_motion_traj(motion_lib)

        self._vis_task = None
        self._plotter = None
        if plot_env0:
            self._build_plotter()

    def _build_plotter(self):
        self._vis_task = Draw3DSkeletonMotion(task_name="motion", skeleton_motion=self._motions_plot[0], frame_index=0)
        self._plotter = Matplotlib3DPlotter(self._vis_task)

    def _build_motion_traj(self, motion_lib: MotionLib):
        self._dof_body_ids = motion_lib.dof_body_ids
        self._dof_offsets = motion_lib.dof_offsets
        self._num_dof = len(self._dof_body_ids)
        self._key_body_ids = motion_lib.key_body_ids
        self._motions = motion_lib.motions
        self._motions_plot = motion_lib.motions_cpu

        frames = []
        motion_start_idx = 0
        self._motion_start_idx = []
        for m in self._motions:
            frame, length = motion_lib_to_matcher(m.global_translation, m.global_rotation, m.global_root_velocity,
                                                  m.global_root_angular_velocity, m.dof_pos, m.dof_vels,
                                                  self._key_body_ids, self._traj_len)
            frames.append(frame)
            self._motion_start_idx.append(motion_start_idx)
            motion_start_idx += length

        self._stacked_motion_traj = torch.cat(frames, dim=0).unsqueeze(0)  # dim0: 1, dim1: frame, dim2: feature
        self._motion_start_idx = torch.tensor(self._motion_start_idx, device=self._device)

    def match(self, obs: torch.Tensor) -> torch.Tensor:
        # dim0: num_env, dim1: feature
        assert len(obs.shape) == 2, f"Only the two observation is allowed. But the shape is {obs.shape}."
        assert (obs.shape[1] == self._stacked_motion_traj.shape[1],
                f"matcher obs size({obs.shape[1]}) != motion_traj size({self._stacked_motion_traj.shape[1]})")

        min_dist_stack_idx = self._min_dist_stack_idx(obs)
        motion_idx = self._frame_idx_to_motion_idx(min_dist_stack_idx)
        if self._plotter is not None:
            frame_idx = min_dist_stack_idx[0] - self._motion_start_idx[motion_idx[0]]
            self._update_plotter(motion_idx[0], frame_idx)
        return motion_idx

    def _min_dist_stack_idx(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.unsqueeze(1)  # dim0: num_env, dim1: 1, dim2: feature
        distance = torch.sum((self._stacked_motion_traj - obs) ** 2, dim=-1)  # dim0: num_env, dim1: frame
        return torch.argmin(distance, dim=-1)

    def _frame_idx_to_motion_idx(self, frame_idx: torch.Tensor) -> torch.Tensor:
        return torch.searchsorted(self._motion_start_idx, frame_idx, right=True) - 1

    def _update_plotter(self, motion_idx, frame_idx):
        self._vis_task.update(skeleton_motion=self._motions_plot[motion_idx], reset_trail=True, frame_index=frame_idx)
        self._plotter.update()


@torch.jit.script
def motion_lib_to_matcher(gts: torch.Tensor, grs: torch.Tensor, grvs: torch.Tensor, gravs: torch.Tensor,
                          dps: torch.Tensor, dvs: torch.Tensor, key_idx: torch.Tensor, traj_len: int
                          ) -> Tuple[torch.Tensor, int]:
    # returns: root_h, local_root_vel/anVel, dof_pos, dof_vel, local_keypoint_pos
    # dim0: frame
    # dim1:    1 +     3*2 +                 31[28] + 31[28] + 3 * 6[4]          = 87[75]
    assert gts.shape[0] > traj_len, f"Frame length({gts.shape[0]}) is shorter than traj_len({traj_len})."

    root_h = gts[:, 0:1, 2]

    inv_heading_rot = calc_heading_quat_inv(grs[:, 0])
    local_root_vel = quat_rotate(inv_heading_rot, grvs)
    local_root_anVel = quat_rotate(inv_heading_rot, gravs)

    dof_pos = dps
    dof_vel = dvs

    local_body_pos = gts - gts[:, 0:1]
    inv_heading_rot_exp = inv_heading_rot.unsqueeze(1).repeat(1, local_body_pos.shape[1], 1)
    l_shape = local_body_pos.shape
    local_body_pos = quat_rotate(inv_heading_rot_exp.view(-1, inv_heading_rot_exp.shape[-1]),
                                 local_body_pos.view(-1, l_shape[-1])).view(l_shape[0], l_shape[1], l_shape[2])
    local_keypoint_pos = local_body_pos[:, key_idx].view(local_body_pos.shape[0], -1)

    motion_frames = torch.cat((root_h, local_root_vel, local_root_anVel, dof_pos, dof_vel, local_keypoint_pos),
                              dim=-1)

    traj_frames = []
    fifo = TensorHistoryFIFO(traj_len)
    for i in range(motion_frames.shape[0]):
        fifo.push(motion_frames[i].unsqueeze(0))
        traj_frames.append(fifo.history)

    return torch.cat(traj_frames, dim=0), motion_frames.shape[0]
