# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
from random import random

import env.tasks.humanoid as humanoid
import env.tasks.humanoid_amp as humanoid_amp
import env.tasks.humanoid_amp_task as humanoid_amp_task
from utils import torch_utils

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *


class HumanoidDoubleReach(humanoid_amp_task.HumanoidAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._tar_speed = cfg["env"]["tarSpeed"]
        self._tar_change_steps_min = cfg["env"]["tarChangeStepsMin"]
        self._tar_change_steps_max = cfg["env"]["tarChangeStepsMax"]
        self._tar_dist_max = cfg["env"]["tarDistMax"]
        self._tar_height_min = cfg["env"]["tarHeightMin"]
        self._tar_height_max = cfg["env"]["tarHeightMax"]
        self._num_reach_bodies = len(cfg["env"]["reachBodyNames"])
        self._num_total_reach_bodies = self._num_reach_bodies * cfg["env"]["numEnvs"]

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self._tar_change_steps = torch.zeros([self.num_envs], device=self.device, dtype=torch.int64)
        self._tar_poses = torch.zeros([self.num_envs, self._num_reach_bodies, 3], device=self.device, dtype=torch.float)

        reach_body_names = cfg["env"]["reachBodyNames"]
        self._reach_body_ids = self._build_reach_body_id_tensor(
            self.envs[0], self.humanoid_handles[0], reach_body_names)

        if not self.headless:
            self._marker_actor_ids = self._build_marker_state_tensors()

        return

    def get_task_obs_size(self):
        obs_size = 0
        if self._enable_task_obs:
            obs_size = 3 * self._num_reach_bodies
        return obs_size

    def _update_marker(self):
        self._marker_poses[..., :] = self._tar_poses
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(self._marker_actor_ids),
                                                     len(self._marker_actor_ids))
        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        if not self.headless:
            self._marker_handles = []
            self._load_marker_asset()

        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _load_marker_asset(self):
        asset_root = "ase/data/assets/mjcf/"
        asset_file = "location_marker.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._marker_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)

        if not self.headless:
            self._build_markers(env_id, env_ptr)

        return

    def _build_markers(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 2
        segmentation_id = 0

        default_pose = gymapi.Transform()

        for i in range(self._num_reach_bodies):
            marker_handle = self.gym.create_actor(
                env_ptr, self._marker_asset, default_pose, f"marker{i}", col_group, col_filter, segmentation_id)
            self.gym.set_rigid_body_color(
                env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(random(), random(), random()))
            self._marker_handles.append(marker_handle)

        return

    def _build_marker_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        self._marker_states = self._root_states.view(
            self.num_envs, num_actors, self._root_states.shape[-1])[..., 1:, :]  # [envs, bodies, state]
        self._marker_poses = self._marker_states[..., :3]  # [envs, bodies, 3]

        marker_actor_ids = []
        for i in range(self._num_reach_bodies):
            marker_actor_ids.append(self._humanoid_actor_ids + i + 1)

        return torch.stack(marker_actor_ids, dim=1)

    def _build_reach_body_id_tensor(self, env_ptr, actor_handle, body_names):
        body_ids = []
        for name in body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, name)
            assert (body_id != -1)
            body_id = to_torch(body_id, device=self.device, dtype=torch.long)
            body_ids.append(body_id.unsqueeze(0))
        return torch.cat(body_ids, dim=0)

    def _update_task(self):
        reset_task_mask = self.progress_buf >= self._tar_change_steps
        rest_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
        if len(rest_env_ids) > 0:
            self._reset_task(rest_env_ids)
        return

    def _reset_task(self, env_ids):
        n = len(env_ids)

        for i in range(self._num_reach_bodies):
            rand_pos = torch.rand([n, 3], device=self.device)
            rand_pos[..., 0:2] = self._tar_dist_max * (2.0 * rand_pos[..., 0:2] - 1.0)
            rand_pos[..., 2] = (self._tar_height_max - self._tar_height_min) * rand_pos[..., 2] + self._tar_height_min
            self._tar_poses[env_ids, i, :] = rand_pos
        change_steps = torch.randint(low=self._tar_change_steps_min, high=self._tar_change_steps_max,
                                     size=(n,), device=self.device, dtype=torch.int64)
        self._tar_change_steps[env_ids] = self.progress_buf[env_ids] + change_steps
        return

    def _compute_task_obs(self, env_ids=None):
        if env_ids is None:
            root_states = self._humanoid_root_states
            tar_poses = self._tar_poses
        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_poses = self._tar_poses[env_ids]

        obs = compute_location_observations(root_states, tar_poses)
        return obs.view(obs.shape[0], -1)  # [envs, 3 * num_reach_bodies]

    def _compute_reward(self, actions):
        reach_body_poses = self._rigid_body_pos[:, self._reach_body_ids, :]
        root_rot = self._humanoid_root_states[..., 3:7]
        self.rew_buf[:] = compute_reach_reward(reach_body_poses, root_rot,
                                               self._tar_poses, self._tar_speed,
                                               self.dt)
        return

    def _draw_task(self):
        self._update_marker()

        cols = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

        self.gym.clear_lines(self.viewer)

        starts = self._rigid_body_pos[:, self._reach_body_ids, :]
        ends = self._tar_poses

        verts = torch.cat([starts, ends], dim=-1).cpu().numpy()  # [envs, num_reach_bodies, 6]

        for i, env_ptr in enumerate(self.envs):
            for j in range(verts.shape[1]):
                curr_verts = verts[i, j]
                curr_verts = curr_verts.reshape([1, 6])
                self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)

        return


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_location_observations(root_states, tar_poses):
    # type: (Tensor, Tensor) -> Tensor
    # root_states: [envs, 13]
    # tar_pos: [envs, tar_id(=2), 3]
    root_rot = root_states[:, 3:7]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    obs = torch.empty_like(tar_poses)

    for i in range(tar_poses.size(1)):
        local_tar_pos = quat_rotate(heading_rot, tar_poses[:, i])
        obs[:, i] = local_tar_pos

    return obs


@torch.jit.script
def compute_reach_reward(reach_body_poses, root_rot, tar_poses, tar_speed, dt):
    # type: (Tensor, Tensor, Tensor, float, float) -> Tensor
    # reach_body_poses: [envs, num_reach_bodies, 3]
    # tar_poses: [envs, num_reach_bodies, 3]

    pos_err_scale = 4.0

    pos_diff = tar_poses - reach_body_poses
    pos_errs = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_err = torch.sum(pos_errs, dim=-1)
    pos_reward = torch.exp(-pos_err_scale * pos_err)

    reward = pos_reward

    return reward
