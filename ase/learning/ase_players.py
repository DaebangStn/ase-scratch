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
import time
import matplotlib.pyplot as plt

from isaacgym.torch_utils import *
from rl_games.algos_torch import players

from learning import amp_players
from learning import ase_network_builder
from learning.logger.motionTransition import MotionTransitionLogger
from learning.logger.matcher import Matcher
from utils.buffer import TensorHistoryFIFO
from utils.angle import calc_heading_quat_inv, quat_rotate


class ASEPlayer(amp_players.AMPPlayerContinuous):
    def __init__(self, config):
        self._latent_dim = config['latent_dim']
        self._latent_steps_min = config.get('latent_steps_min', np.inf)
        self._latent_steps_max = config.get('latent_steps_max', np.inf)

        self._enc_reward_scale = config['enc_reward_scale']

        self._matcher = None
        self._matcher_obs_buf = None
        self._transition_logger = None

        super().__init__(config)
        
        if (hasattr(self, 'env')):
            batch_size = self.env.task.num_envs
        else:
            batch_size = self.env_info['num_envs']
        self._ase_latents = torch.zeros((batch_size, self._latent_dim), dtype=torch.float32,
                                         device=self.device)

        logger_config = self.config.get('logger', None)
        if logger_config is not None:
            log_latent = logger_config.get('latent_motion_id', False)
            motion_transition = logger_config.get('motion_transition', False)
            if log_latent or motion_transition:
                show_matcher_out = logger_config.get('show_matcher_out', False)

                self._matcher_obs_buf = TensorHistoryFIFO(self.env.task.amp_obs_size)

                plt.switch_backend('TkAgg')  # Since pycharm IDE embeds matplotlib, it is necessary to switch backend
                self._matcher = Matcher(self.env.task.motion_lib, self.env.task.amp_obs_size, self.env.task.dt, self.device,
                                        show_matcher_out)
                self._transition_logger = (
                    MotionTransitionLogger(logger_config['filename'], time.strftime("%Y%m%d-%H%M%S"),
                                           self.env.task.num_envs))

        return

    def run(self):
        self._reset_latent_step_count()
        super().run()
        return

    def get_action(self, obs_dict, is_determenistic=False):
        self._update_latents()

        obs = obs_dict['obs']
        if len(obs.size()) == len(self.obs_shape):
            obs = obs.unsqueeze(0)
        obs = self._preproc_obs(obs)
        ase_latents = self._ase_latents

        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs,
            'rnn_states' : self.states,
            'ase_latents': ase_latents
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if is_determenistic:
            current_action = mu
        else:
            current_action = action
        current_action = current_action.detach()
        return  players.rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))

    def env_step(self, env, actions):
        obs, rew, done, info = super().env_step(env, actions)
        if self._transition_logger:
            self._matcher_obs_buf.push(self.env.task.matcher_obs())
            motion_id = self._matcher.match(self._matcher_obs_buf.history)
            self._transition_logger.log(motion_id)
        return obs, rew, done, info

    def env_reset(self, env_ids=None):
        obs = super().env_reset(env_ids)
        self._reset_latents(env_ids)
        if self._transition_logger:
            resets = torch.zeros(self.env.task.num_envs, dtype=torch.bool, device=self.device)
            resets[env_ids] = True
            self._matcher_obs_buf.push_on_reset(self.env.task.matcher_obs(), resets)
        return obs
    
    def _build_net_config(self):
        config = super()._build_net_config()
        config['ase_latent_shape'] = (self._latent_dim,)
        return config
    
    def _reset_latents(self, done_env_ids=None):
        if (done_env_ids is None):
            num_envs = self.env.task.num_envs
            done_env_ids = to_torch(np.arange(num_envs), dtype=torch.long, device=self.device)

        rand_vals = self.model.a2c_network.sample_latents(len(done_env_ids))
        self._ase_latents[done_env_ids] = rand_vals
        self._change_char_color(done_env_ids)

        return

    def _update_latents(self):
        if self._latent_step_count <= 0:
            self._reset_latents()
            self._reset_latent_step_count()

            num_envs = self.env.task.num_envs
            env_ids = to_torch(np.arange(num_envs), dtype=torch.long, device=self.device)
            if self.env.task.viewer:
                print("Sampling new amp latents------------------------------")
                self._change_char_color(env_ids)
            if self._transition_logger:
                self._transition_logger.update_z(env_ids)
        else:
            self._latent_step_count -= 1
        return
    
    def _reset_latent_step_count(self):
        self._latent_step_count = np.random.randint(self._latent_steps_min, self._latent_steps_max)
        return

    def _calc_amp_rewards(self, amp_obs, ase_latents):
        disc_r = self._calc_disc_rewards(amp_obs)
        enc_r = self._calc_enc_rewards(amp_obs, ase_latents)
        output = {
            'disc_rewards': disc_r,
            'enc_rewards': enc_r
        }
        return output
    
    def _calc_enc_rewards(self, amp_obs, ase_latents):
        with torch.no_grad():
            enc_pred = self._eval_enc(amp_obs)
            err = self._calc_enc_error(enc_pred, ase_latents)
            enc_r = torch.clamp_min(-err, 0.0)
            enc_r *= self._enc_reward_scale

        return enc_r
    
    def _calc_enc_error(self, enc_pred, ase_latent):
        err = enc_pred * ase_latent
        err = -torch.sum(err, dim=-1, keepdim=True)
        return err
    
    def _eval_enc(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_enc(proc_amp_obs)

    def _amp_debug(self, info):
        with torch.no_grad():
            amp_obs = info['amp_obs']
            amp_obs = amp_obs
            ase_latents = self._ase_latents
            disc_pred = self._eval_disc(amp_obs)
            amp_rewards = self._calc_amp_rewards(amp_obs, ase_latents)
            disc_reward = amp_rewards['disc_rewards']
            enc_reward = amp_rewards['enc_rewards']

            disc_pred = disc_pred.detach().cpu().numpy()[0, 0]
            disc_reward = disc_reward.cpu().numpy()[0, 0]
            enc_reward = enc_reward.cpu().numpy()[0, 0]
            print("disc_pred: ", disc_pred, disc_reward, enc_reward)
        return

    def _change_char_color(self, env_ids):
        base_col = np.array([0.4, 0.4, 0.4])
        range_col = np.array([0.0706, 0.149, 0.2863])
        range_sum = np.linalg.norm(range_col)

        rand_col = np.random.uniform(0.0, 1.0, size=3)
        rand_col = range_sum * rand_col / np.linalg.norm(rand_col)
        rand_col += base_col
        self.env.task.set_char_color(rand_col, env_ids)
        return
