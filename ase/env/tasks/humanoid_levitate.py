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

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *

import env.tasks.humanoid_amp as humanoid_amp
import env.tasks.humanoid_amp_getup as humanoid_amp_getup
import env.tasks.humanoid_strike as humanoid_strike
import env.tasks.humanoid_location as humanoid_location
from utils import torch_utils


class HumanoidLevitate(humanoid_amp.HumanoidAMP):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self._levi_h_min = 0.25
        self._levi_h_max = 2.0
        return

    def _reset_default(self, env_ids):
        random_heights = (torch.rand(env_ids.shape, device=self.device, dtype=torch.float32) *
                          (self._levi_h_max - self._levi_h_min) + self._levi_h_min)
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._humanoid_root_states[env_ids, 2] += random_heights
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        self._reset_default_env_ids = env_ids
        return

#####################################################################
###=========================jit functions=========================###
#####################################################################

