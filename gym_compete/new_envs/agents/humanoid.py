from .agent import Agent
from gymnasium.spaces import Box
import numpy as np
import os


def mass_center(mass, xpos):
    # This is a bug of openai multiagent-competition release.
    # It should keep the same dimension when array broadcast.
    # Modified by git @KJaebye.
    mass = mass[:, np.newaxis]
    return (np.sum(mass * xpos, 0) / np.sum(mass))[:2]


class Humanoid(Agent):

    def __init__(self, agent_id, xml_path=None, n_agents=2, **kwargs):
        if xml_path is None:
            xml_path = os.path.join(os.path.dirname(__file__), "assets", "humanoid_body.xml")
        super(Humanoid, self).__init__(agent_id, xml_path, n_agents, **kwargs)
        self.team = 'walker'

    def set_goal(self, goal):
        self.GOAL = goal
        self.move_left = False
        if self.get_qpos()[0] > 0:
            self.move_left = True

    def quat2euler(self, quat):
        w, x, y, z = quat
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        X = np.arctan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1, 1)
        Y = np.arcsin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        Z = np.arctan2(t3, t4)
        return np.array([X, Y, Z])

    def before_step(self):
        self._pos_before = mass_center(self.get_body_mass(), self.get_xipos())

        torso_quat = self.env.data.qpos[3:7]
        self.torso_euler = self.quat2euler(torso_quat)
        #degree
        # self.torso_euler = np.rad2deg(self.torso_euler)
        # print("Torso Euler:", self.torso_euler)
        # self.heading = [np.cos(self.torso_euler[2]), np.sin(self.torso_euler[2])]
        self.dist_before = np.linalg.norm(self.GOAL-self._pos_before)

    def after_step(self, action):
        pos_after = mass_center(self.get_body_mass(), self.get_xipos())
        # forward_reward = 1.25 * (pos_after - self._pos_before) / self.env.model.opt.timestep
        # forward_reward = .25 * (pos_after - self._pos_before) / self.env.model.opt.timestep
        heading = np.array([np.cos(self.torso_euler[2]), np.sin(self.torso_euler[2])])
        forward_reward = 1.25 * np.dot(heading, pos_after - self._pos_before) / self.env.model.opt.timestep

        dist_after = np.linalg.norm(self.GOAL-pos_after)
        dist_reward = (self.dist_before - dist_after) / self.env.model.opt.timestep

        # if self.move_left:
        #     forward_reward *= -1
        ctrl_cost = .1 * np.square(action).sum()
        cfrc_ext = self.get_cfrc_ext()
        contact_cost = .5e-6 * np.square(cfrc_ext).sum()
        contact_cost = min(contact_cost, 10)
        qpos = self.get_qpos()
        survive = 5.0
        reward = forward_reward - ctrl_cost - contact_cost + survive# + dist_reward

        # reward_goal = - np.abs(qpos[0].item() - self.GOAL)
        # reward += reward_goal

        reward_info = dict()
        reward_info['reward_forward'] = forward_reward
        reward_info['reward_ctrl'] = ctrl_cost
        reward_info['reward_contact'] = contact_cost
        reward_info['reward_survive'] = survive
        reward_info['reward_dist'] = dist_reward
        # if self.team == 'walker':
        #     reward_info['reward_goal_dist'] = reward_goal
        reward_info['reward_dense'] = reward

        # done = not agent_standing
        terminated = bool(qpos[2] < 1. or qpos[2] > 2.)

        return reward, terminated, reward_info


    def _get_obs(self):
        '''
        Return agent's observations
        '''
        my_pos = self.get_qpos()
        other_pos = self.get_other_qpos()
        vel = self.get_qvel()
        cfrc_ext = np.clip(self.get_cfrc_ext(), -1, 1)
        cvel = self.get_cvel()
        cinert = self.get_cinert()
        qfrc_actuator = self.get_qfrc_actuator()

        obs = np.concatenate(
            [my_pos.flat, vel.flat,
             cinert.flat, cvel.flat,
             qfrc_actuator.flat, cfrc_ext.flat,
             other_pos.flat, self.GOAL]
        )
        assert np.isfinite(obs).all(), "Humanoid observation is not finite!!"
        return obs

    def _get_obs_relative(self):
        '''
        Return agent's observations, positions are relative
        '''
        qpos = self.get_qpos()
        my_pos = qpos[2:]
        other_agents_qpos = self.get_other_agent_qpos()
        all_other_qpos = []
        for i in range(self.n_agents):
            if i == self.id: continue
            other_qpos = other_agents_qpos[i]
            other_relative_xy = other_qpos[:2] - qpos[:2]
            other_qpos = np.concatenate([other_relative_xy.flat, other_qpos[2:].flat], axis=0)
            all_other_qpos.append(other_qpos)
        all_other_qpos = np.concatenate(all_other_qpos)

        vel = self.get_qvel()
        cfrc_ext = np.clip(self.get_cfrc_ext(), -1, 1)
        cvel = self.get_cvel()
        cinert = self.get_cinert()
        qfrc_actuator = self.get_qfrc_actuator()

        obs = np.concatenate(
            [my_pos.flat, vel.flat,
             cinert.flat, cvel.flat,
             qfrc_actuator.flat, cfrc_ext.flat,
             all_other_qpos.flat]
        )
        assert np.isfinite(obs).all(), "Humanoid observation is not finite!!"
        return obs

    def set_observation_space(self):
        obs = self._get_obs()
        self.obs_dim = obs.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = Box(low, high)

    def reached_goal(self):
        # if self.n_agents == 1: return False
        # xpos = self.get_body_com('torso')[0]
        # if self.GOAL > 0 and xpos > self.GOAL:
        #     return True
        # elif self.GOAL < 0 and xpos < self.GOAL:
        #     return True
        if self.dist_before < 1.0:
            return True
        return False

    def reset_agent(self):
        if self.n_agents > 1:
            xpos = self.get_qpos()[0]
            if xpos * self.GOAL > 0 :
                self.set_goal(-self.GOAL)
            if xpos > 0:
                self.move_left = True
            else:
                self.move_left = False
