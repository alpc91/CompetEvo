from runner.base_runner import BaseRunner
from custom.learners.learner import Learner
from custom.learners.sampler import Sampler
from custom.learners.evo_sampler import EvoSampler
from custom.learners.evo_learner import EvoLearner
from custom.learners.dev_learner import DevLearner
from custom.learners.dev_sampler import DevSampler
from custom.utils.logger import MaLoggerRL
from lib.rl.core.trajbatch import MaTrajBatch, MaTrajBatchDisc
from lib.utils.torch import *
from lib.utils.memory import Memory

import csv
import time
import math
import os
import platform
import pickle
import multiprocessing
import gymnasium as gym
if platform.system() != "Linux":
    from multiprocessing import set_start_method
    set_start_method("fork")
os.environ["OMP_NUM_THREADS"] = "1"

from operator import add
from functools import reduce
import collections

import random

def write_csv_data(csv_file_path, data: list):
    with open(csv_file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)

def tensorfy(np_list, device=torch.device('cpu')):
    if isinstance(np_list[0], list):
        return [[torch.tensor(x).to(device) for i, x in enumerate(y)] for y in np_list]
    else:
        return [torch.tensor(y).to(device) for y in np_list]
    
def mix_tensorfy(np_list, device=torch.device('cpu')):
    res = []
    for l in np_list:
        if isinstance(l, list):
            res.append([torch.tensor(x).to(device) for i, x in enumerate(l)])
        else:
            res.append(torch.tensor(l).to(device))
    return res
    

class MultiEvoAgentRunner(BaseRunner):
    def __init__(self, cfg, logger, dtype, device, num_threads=1, training=True, ckpt_dir=None, ckpt=0) -> None:
        super().__init__(cfg, logger, dtype, device, num_threads=num_threads, training=training, ckpt_dir=ckpt_dir, ckpt=ckpt)
        self.agent_num = self.learners.__len__()

        self.logger_cls = MaLoggerRL
        self.traj_cls = MaTrajBatchDisc

        self.logger_kwargs = dict()

        self.end_reward = False

    def setup_learner(self):
        """ Learners are corresponding to agents. """
        self.learners = {}
        for i, agent in self.env.agents.items():
            if hasattr(agent, "flag") and agent.flag == "evo":
                self.learners[i] = EvoLearner(self.cfg, self.dtype, self.device, self.env.agents[i])
            elif hasattr(agent, "flag") and agent.flag == "dev":
                self.learners[i] = DevLearner(self.cfg, self.dtype, self.device, self.env.agents[i])
            else:
                self.learners[i] = Learner(self.cfg, self.dtype, self.device, self.env.agents[i])

    def optimize_policy(self):
        epoch = self.epoch
        """generate multiple trajectories that reach the minimum batch_size"""
        self.logger.info('#------------------------ Iteration {} --------------------------#'.format(epoch)) # actually this is iteration
        t0 = time.time()

        """sampling data"""
        batches, logs, t_win_rate = self.sample(self.cfg.min_batch_size)
        t1 = time.time()
        self.logger.info(
            "Sampling {} steps by {} slaves, spending {:.2f} s.".format(
            self.cfg.min_batch_size, self.num_threads, t1-t0)
        )
        
        
        """updating policy"""
        policy_losses = {}
        value_losses = {}
        for i, learner in self.learners.items():
            policy_loss, value_loss = learner.update_params(batches[i])
            policy_losses[i] = policy_loss
            value_losses[i] = value_loss
        t2 = time.time()
        self.logger.info("Policy update, spending: {:.2f} s.".format(t2-t1))

        """evaluate policy"""
        _, log_evals, win_rate = self.sample(self.cfg.eval_batch_size, mean_action=True, nthreads=10, eval=True)
        t3 = time.time()
        self.logger.info("Evaluation time: {:.2f} s.".format(t3-t2))

        # """evaluate best policy"""
        # _, log_best_evals, b_win_rate = self.sample(self.cfg.eval_batch_size, mean_action=True, nthreads=10, best = True)
        # t4 = time.time()
        # self.logger.info("Evaluation time: {:.2f} s.".format(t4-t3))

        info = {
            'logs': logs, 'log_evals': log_evals, 
            # 'log_best_evals': log_best_evals,
            't_win_rate': t_win_rate,
            'win_rate': win_rate,
            # 'b_win_rate': b_win_rate,
            'policy_losses': policy_losses, 'value_losses': value_losses
        }
        self.logger.info('Total time: {:10.2f} min'.format((t3 - self.t_start)/60))
        return info
    
    def log_optimize_policy(self, info):
        epoch = self.epoch
        cfg = self.cfg
        logs, log_evals, t_win_rate, win_rate,  policy_losses, value_losses= info['logs'], info['log_evals'], info['t_win_rate'], info['win_rate'], info['policy_losses'], info['value_losses']
        # log_best_evals, b_win_rate, info['b_win_rate'],  info['log_best_evals'],
        logger, writer = self.logger, self.writer

        # print("0:", logs[0].total_reward, logs[0].num_episodes, logs[0].avg_episode_reward)
        # print("1:", logs[1].total_reward, logs[1].num_episodes, logs[1].avg_episode_reward)
            
        for i, learner in self.learners.items():
            logger.info("Agent_{} gets policy loss: {:.2f}.".format(i, policy_losses[i]))
            logger.info("Agent_{} gets value loss: {:.2f}.".format(i, value_losses[i]))

            logger.info("Agent_{} gets train_num_episodes: {:.2f}.".format(i, logs[i].num_episodes))
            logger.info("Agent_{} gets train_num_steps: {:.2f}.".format(i, logs[i].num_steps))
            logger.info("Agent_{} gets train_avg_episode_len: {:.2f}.".format(i, logs[i].avg_episode_len))
            logger.info("Agent_{} gets train_avg_episode_reward: {:.2f}.".format(i, logs[i].avg_episode_reward))
            logger.info("Agent_{} gets train_min_dist: {:.2f}.".format(i, logs[i].min_dist))
            logger.info("Agent_{} gets train_max_dist: {:.2f}.".format(i, logs[i].max_dist))
            logger.info("Agent_{} gets train_win rate: {:.2f}.".format(i, t_win_rate[i]))

            logger.info("Agent_{} gets eval_num_episodes: {:.2f}.".format(i, log_evals[i].num_episodes))
            logger.info("Agent_{} gets eval_num_steps: {:.2f}.".format(i, log_evals[i].num_steps))
            logger.info("Agent_{} gets eval eval_avg_episode_len: {:.2f}.".format(i, log_evals[i].avg_episode_len))
            logger.info("Agent_{} gets eval eval_avg_episode_reward: {:.2f}.".format(i, log_evals[i].avg_episode_reward))
            logger.info("Agent_{} gets eval_min_dist: {:.2f}.".format(i, log_evals[i].min_dist))
            logger.info("Agent_{} gets eval_max_dist: {:.2f}.".format(i, log_evals[i].max_dist))
            logger.info("Agent_{} gets eval_win rate: {:.2f}.".format(i, win_rate[i]))

            # logger.info("Agent_{} gets best_eval_num_episodes: {:.2f}.".format(i, log_best_evals[i].num_episodes))
            # logger.info("Agent_{} gets best_eval_num_steps: {:.2f}.".format(i, log_best_evals[i].num_steps))
            # logger.info("Agent_{} gets best_eval_avg_episode_len: {:.2f}.".format(i, log_best_evals[i].avg_episode_len))
            # logger.info("Agent_{} gets best_eval_avg_episode_reward: {:.2f}.".format(i, log_best_evals[i].avg_episode_reward))
            # logger.info("Agent_{} gets best_eval_win rate: {:.2f}.".format(i, b_win_rate[i]))


            if log_evals[i].avg_episode_reward > learner.best_reward or win_rate[i] > learner.best_win_rate:
                learner.best_reward = log_evals[i].avg_episode_reward
                learner.best_win_rate = win_rate[i]
                learner.save_best_flag = True
            else:
                learner.save_best_flag = False

            # writer.add_scalar('train_R_avg_{}'.format(i), logs[i].avg_reward, epoch)
            writer.add_scalar('train_R_eps_avg_{}'.format(i), logs[i].avg_episode_reward, epoch)
            writer.add_scalar('eval_R_eps_avg_{}'.format(i), log_evals[i].avg_episode_reward, epoch)
            # writer.add_scalar('best_eval_R_eps_avg_{}'.format(i), log_best_evals[i].avg_episode_reward, epoch)
            # writer.add_scalar('eval_R_avg_{}'.format(i), log_evals[i].avg_reward, epoch)

            # logging win rate
            writer.add_scalar("train_win_rate_{}".format(i), t_win_rate[i], epoch)
            # logging win rate
            writer.add_scalar("eval_win_rate_{}".format(i), win_rate[i], epoch)
            # logging win rate
            # writer.add_scalar("best_eval_win_rate_{}".format(i), b_win_rate[i], epoch)

            # eps num_episodes
            writer.add_scalar("train_num_episodes", logs[i].num_episodes, epoch)
            # eps num_episodes
            writer.add_scalar("eval_num_episodes", log_evals[i].num_episodes, epoch)
            # eps num_episodes
            # writer.add_scalar("best_eval_num_episodes", log_best_evals[i].num_episodes, epoch)

            # eps num_episodes
            writer.add_scalar("train_num_steps", logs[i].num_steps, epoch)
            # eps num_episodes
            writer.add_scalar("eval_num_steps", log_evals[i].num_steps, epoch)
            # eps num_episodes
            # writer.add_scalar("best_eval_num_steps", log_best_evals[i].num_steps, epoch)


            # eps len
            writer.add_scalar("train_episode_length", logs[i].avg_episode_len, epoch)
            # eps len
            writer.add_scalar("eval_episode_length", log_evals[i].avg_episode_len, epoch)
            # eps len
            # writer.add_scalar("best_eval_episode_length", log_best_evals[i].avg_episode_len, epoch)

            # min dist
            writer.add_scalar("train_min_dist", logs[i].min_dist, epoch)
            writer.add_scalar("eval_min_dist", log_evals[i].min_dist, epoch)

            # max dist
            writer.add_scalar("train_max_dist", logs[i].max_dist, epoch)
            writer.add_scalar("eval_max_dist", log_evals[i].max_dist, epoch)

            
            # policy loss
            writer.add_scalar("policy_loss_{}".format(i), policy_losses[i], epoch)
            # value loss
            writer.add_scalar("value_loss_{}".format(i), value_losses[i], epoch)
            
    
    def optimize(self, epoch):
        self.epoch = epoch
        # set annealing params
        for i in self.learners:
            self.learners[i].pre_epoch_update(epoch)
        info = self.optimize_policy()
        self.log_optimize_policy(info)

    def push_memory(self, memories, states, actions, masks, next_states, rewards, exps):
        for i, memory in enumerate(memories):
            memory.push(states[i], actions[i], masks[i], next_states[i], rewards[i], exps[i])
    
    def custom_reward(self, infos):
        """ Exploration curriculum: 
            set linear annealing factor alpha to balance parse and dense rewards. 
        """
        if hasattr(self, "epoch"):
            epoch = self.epoch
        else:
            # only for display
            epoch = 1000
        termination_epoch = self.cfg.termination_epoch
        alpha = max((termination_epoch - epoch) / termination_epoch, 0)
        c_rew = []
        for i, info in enumerate(infos):
            parse_rew = info['reward_parse'] # goal_rew is parse rewarding
            dense_rew = info['reward_dense'] # move_rew is dense rewarding
            rew = alpha * dense_rew + (1-alpha) * parse_rew
            c_rew.append(rew)
        return tuple(c_rew), infos

    def sample_worker(self, pid, queue, min_batch_size, mean_action, render, randomstate, idx=None, eval=False):
        self.seed_worker(pid)
        design_params = {0: [], 1: []}
        
        # define multi-agent logger
        ma_logger = self.logger_cls(self.agent_num, **self.logger_kwargs).loggers
        # total score record: [agent_0_win_times, agent_1_win_times, games_num]
        total_score = [0 for _ in range(self.agent_num)]
        total_score.append(0)
        # define multi-agent memories
        ma_memory = []
        for i in range(self.agent_num): ma_memory.append(Memory())

        while ma_logger[0].num_steps < min_batch_size:
            # sample random opponent old policies before every rollout
            samplers = {}
            for i in range(self.agent_num):
                if hasattr(self.env.agents[i], 'flag') and self.env.agents[i].flag == "evo":
                    samplers[i] = EvoSampler(self.cfg, self.dtype, 'cpu', self.env.agents[i])
                elif hasattr(self.env.agents[i], "flag") and self.env.agents[i].flag == "dev":
                    samplers[i] = DevSampler(self.cfg, self.dtype, 'cpu', self.env.agents[i])
                else:
                    samplers[i] = Sampler(self.cfg, self.dtype, 'cpu', self.env.agents[i])
                samplers[i].policy_net.load_state_dict(self.learners[i].policy_net.state_dict())

            # sample random opponent old policies before every rollout
            if eval or self.agent_num<2 or self.epoch == 0: #not self.cfg.use_opponent_sample or mean_action 
                ckpt = self.epoch
                # if best:
                #     ckpt = "best"
                #     try:
                #         # get opp/ego ckpt modeal
                #         opp_cp_path = '%s/%s/%s.p' % (self.model_dir, "agent_"+str(0), ckpt)
                #         with open(opp_cp_path, "rb") as f:
                #             opp_model_cp = pickle.load(f)
                #             samplers[0].load_ckpt(opp_model_cp)

                #         # get ego ckpt modeal
                #         ego_cp_path = '%s/%s/%s.p' % (self.model_dir, "agent_"+str(1), ckpt)
                #         with open(ego_cp_path, "rb") as f:
                #             ego_model_cp = pickle.load(f)
                #             samplers[1].load_ckpt(ego_model_cp)
                #     except:
                #         pass
            else:
                assert idx is not None
                """set sampling policy for opponent"""
                start = math.floor(self.epoch * self.cfg.delta)
                start = start if start > 1 else 1
                end = self.epoch
                ckpt = randomstate.randint(start, end+1) if start!=end else end

                # get opp ckpt modeal
                opp_cp_path = '%s/%s/epoch_%04d.p' % (self.model_dir, "agent_"+str(1-idx), ckpt)
                with open(opp_cp_path, "rb") as f:
                    opp_model_cp = pickle.load(f)
                    samplers[1-idx].load_ckpt(opp_model_cp)

                # # get ego ckpt modeal
                # ego_cp_path = '%s/%s/epoch_%04d.p' % (self.model_dir, "agent_"+str(idx), self.epoch)
                # with open(ego_cp_path, "rb") as f:
                #     ego_model_cp = pickle.load(f)
                #     samplers[idx].load_ckpt(ego_model_cp)

            states, info = self.env.reset(symmetric=eval or self.cfg.symmetric)
            # normalize states
            for i, sampler in samplers.items():
                if sampler.running_state is not None:
                    states[i] = sampler.running_state(states[i])
                ma_logger[i].start_episode(self.env)
            
            for t in range(10000):
                state_var = mix_tensorfy(states)
                use_mean_action = mean_action or torch.bernoulli(torch.tensor([1 - self.noise_rate])).item()
                # select actions
                actions = []
                
                for i, sampler in samplers.items():
                    if hasattr(sampler, 'flag') and self.env.agents[i].flag == "evo":
                        actions.append(sampler.policy_net.select_action([state_var[i]], use_mean_action).squeeze().numpy().astype(np.float64))
                    elif hasattr(sampler, 'flag') and self.env.agents[i].flag == "dev":
                        actions.append(sampler.policy_net.select_action([state_var[i]], use_mean_action).squeeze().numpy().astype(np.float64))
                    else:
                        actions.append(sampler.policy_net.select_action(state_var[i], use_mean_action).squeeze().numpy().astype(np.float64))
                
                next_states, env_rewards, terminateds, truncated, infos = self.env.step(actions)
                
                # normalize states
                for i, sampler in samplers.items():
                    if sampler.running_state is not None:
                        next_states[i] = sampler.running_state(next_states[i])

                if not mean_action and "design_params" in infos[0]:
                    for i, info in enumerate(infos):
                        d = info["design_params"]
                        design_params[i].append(d)

                # use custom or env reward
                if self.cfg.use_exploration_curriculum:
                    assert hasattr(self, 'custom_reward')
                    c_rewards, c_infos = self.custom_reward(infos)
                    rewards = c_rewards
                else:
                    c_rewards, c_infos = [0.0 for _ in samplers], [np.array([0.0]) for _ in samplers]
                    rewards = env_rewards
                # add end reward
                if self.end_reward and infos.get('end', False):
                    rewards += self.env.end_rewards
                
                # logging (logging the original rewards)
                for i, logger in enumerate(ma_logger):
                    logger.step(self.env, rewards[i], c_rewards[i], c_infos[i], infos[i])

                # normalize rewards and train use this value
                if self.cfg.use_reward_scaling:
                    assert samplers[0].reward_scaling
                    rewards = list(rewards)
                    for i, sampler in samplers.items():
                        rewards[i] = sampler.reward_scaling(rewards[i])
                    rewards = tuple(rewards)

                masks = [1 for _ in samplers]
                if truncated:
                    draw = True
                elif terminateds[0]:
                    for i in samplers:
                        if "winner" in infos[i]:
                            draw = False
                            total_score[i] += 1
                    masks = [0 for _ in samplers]

                exps = [(1-use_mean_action) for _ in samplers]
                self.push_memory(ma_memory, states, actions, masks, next_states, rewards, exps)

                if terminateds[0] or truncated:
                    total_score[-1] += 1
                    break
                states = next_states

            for logger in ma_logger: logger.end_episode(self.env)
        for logger in ma_logger: logger.end_sampling()
        
        if queue is not None:
            queue.put([pid, ma_memory, ma_logger, total_score, design_params])
        else:
            return ma_memory, ma_logger, total_score, design_params

    def sample(self, min_batch_size, mean_action=False, render=False, nthreads=None, eval=False):
        if nthreads is None:
            nthreads = self.num_threads

        t_start = time.time()
        for i, learner in self.learners.items():
            to_test(*learner.sample_modules)

        if eval or self.agent_num<2:#(not self.cfg.use_opponent_sample) or mean_action:
            with to_cpu(*reduce(add, (learner.sample_modules for i, learner in self.learners.items()))):
                with torch.no_grad():
                    thread_batch_size = int(math.floor(min_batch_size / nthreads))
                    torch.set_num_threads(1) # bug occurs due to thread pools copy while forking
                    queue = multiprocessing.Queue()
                    memories = [None] * nthreads
                    loggers = [None] * nthreads
                    total_scores = [None] * nthreads
                    design_params = [None] * nthreads

                    for i in range(nthreads-1):
                        worker_args = (i+1, queue, thread_batch_size, mean_action, render, np.random.RandomState(), None, eval)
                        worker = multiprocessing.Process(target=self.sample_worker, args=worker_args)
                        worker.start()
                    memories[0], loggers[0], total_scores[0], design_params[0] = self.sample_worker(0, None, thread_batch_size, mean_action, render, np.random.RandomState(), None, eval)

                    for i in range(nthreads - 1):
                        pid, worker_memory, worker_logger, total_score, design_param = queue.get()
                        memories[pid] = worker_memory
                        loggers[pid] = worker_logger
                        total_scores[pid] = total_score
                        design_params[pid] = design_param

                    ## design params
                    design_params0 = []
                    design_params1 = []
                    for i in range(nthreads):
                        for l in design_params[i][0]:
                            design_params0.append(l)
                        for l in design_params[i][1]:
                            design_params1.append(l)

                    if len(design_params0) > 10:
                        design_params0 = random.sample(design_params0, 10)
                    if len(design_params1) > 10:
                        design_params1 = random.sample(design_params1, 10)

                    for d in design_params0:
                        file_path = self.run_dir + '/' + '0.csv'
                        d = [self.epoch] + list(d)
                        write_csv_data(file_path, d)
                    for d in design_params1:
                        file_path = self.run_dir + '/' + '1.csv'
                        d = [self.epoch] + list(d)
                        write_csv_data(file_path, d)

                    # merge batch data and log data from multiprocessings
                    ma_buffer = self.traj_cls(memories).buffers
                    ma_logger = self.logger_cls.merge(loggers, **self.logger_kwargs)

                    # win rate
                    total_scores = list(zip(*total_scores))
                    total_scores = [sum(scores) for scores in total_scores]
                    win_rate = [total_scores[0]/total_scores[-1], total_scores[1]/total_scores[-1]]
                    
                    if self.agent_num > 1:
                        self.logger.info("eval total_scores: {}, {}, {}".format(total_scores[0],total_scores[1],total_scores[2]))
                
            for logger in ma_logger: logger.sample_time = time.time() - t_start
            return ma_buffer, ma_logger, win_rate

        else:
            with to_cpu(*reduce(add, (learner.sample_modules for i, learner in self.learners.items()))):
                with torch.no_grad():
                    thread_batch_size = int(math.floor(min_batch_size / nthreads))
                    torch.set_num_threads(1) # bug occurs due to thread pools copy while forking
                    # for agent0
                    queue_0 = multiprocessing.Queue()
                    memories_0 = [None] * nthreads
                    loggers_0 = [None] * nthreads
                    total_scores_0 = [None] * nthreads
                    # for agent1
                    queue_1 = multiprocessing.Queue()
                    memories_1 = [None] * nthreads
                    loggers_1 = [None] * nthreads
                    total_scores_1 = [None] * nthreads

                    design_params_0 = [None] * nthreads
                    design_params_1 = [None] * nthreads

                    for i in range(nthreads-1):
                        worker_args_0 = (i+1, queue_0, thread_batch_size, mean_action, render, np.random.RandomState(), 0)
                        worker_0 = multiprocessing.Process(target=self.sample_worker, args=worker_args_0)
                        worker_0.start()
                        worker_args_1 = (i+1, queue_1, thread_batch_size, mean_action, render, np.random.RandomState(), 1)
                        worker_1 = multiprocessing.Process(target=self.sample_worker, args=worker_args_1)
                        worker_1.start()
                    memories_0[0], loggers_0[0], total_scores_0[0], design_params_0[0] = self.sample_worker(0, None, thread_batch_size, mean_action, render, np.random.RandomState(), 0)
                    memories_1[0], loggers_1[0], total_scores_1[0], design_params_1[0] = self.sample_worker(0, None, thread_batch_size, mean_action, render, np.random.RandomState(), 1)

                    for i in range(nthreads - 1):
                        pid_0, worker_memory_0, worker_logger_0, total_score_0, design_param_0 = queue_0.get()
                        memories_0[pid_0] = worker_memory_0
                        loggers_0[pid_0] = worker_logger_0
                        total_scores_0[pid_0] = total_score_0
                        design_params_0[pid_0] = design_param_0

                        pid_1, worker_memory_1, worker_logger_1, total_score_1, design_param_1 = queue_1.get()
                        memories_1[pid_1] = worker_memory_1
                        loggers_1[pid_1] = worker_logger_1
                        total_scores_1[pid_1] = total_score_1
                        design_params_1[pid_1] = design_param_1
                    
                    design_params0 = []
                    design_params1 = []
                    for i in range(nthreads):
                        for l in design_params_0[i][0]:
                            design_params0.append(l)
                        for l in design_params_1[i][1]:
                            design_params1.append(l)

                    if len(design_params0) > 10:
                        design_params0 = random.sample(design_params0, 10)
                    if len(design_params1) > 10:
                        design_params1 = random.sample(design_params1, 10)

                    for d in design_params0:
                        file_path = self.run_dir + '/' + '0.csv'
                        d = [self.epoch] + list(d)
                        write_csv_data(file_path, d)
                    for d in design_params1:
                        file_path = self.run_dir + '/' + '1.csv'
                        d = [self.epoch] + list(d)
                        write_csv_data(file_path, d)

                    # merge batch data and log data from multiprocessings
                    ma_buffer_0 = self.traj_cls(memories_0).buffers
                    ma_logger_0 = self.logger_cls.merge(loggers_0, **self.logger_kwargs)

                    # merge batch data and log data from multiprocessings
                    ma_buffer_1 = self.traj_cls(memories_1).buffers
                    ma_logger_1 = self.logger_cls.merge(loggers_1, **self.logger_kwargs)

                    # win rate
                    total_scores_0 = list(zip(*total_scores_0))
                    total_scores_0 = [sum(scores) for scores in total_scores_0]
                    total_scores_1 = list(zip(*total_scores_1))
                    total_scores_1 = [sum(scores) for scores in total_scores_1]
                    win_rate = [total_scores_0[0]/total_scores_0[-1], total_scores_1[1]/total_scores_1[-1]]
                    if self.agent_num > 1:
                        self.logger.info("train total_scores_0: {}, {}, {}".format(total_scores_0[0],total_scores_0[1],total_scores_0[2]))
                        self.logger.info("train total_scores_1: {}, {}, {}".format(total_scores_1[0],total_scores_1[1],total_scores_1[2]))

                    # extract corresponding agent data 
                    b = [ma_buffer_0[0], ma_buffer_1[1]]
                    l = [ma_logger_0[0], ma_logger_1[1]]

            for _ in l: _.sample_time = time.time() - t_start
            return b, l, win_rate
    
    def load_checkpoint(self, ckpt_dir, checkpoint):
        assert isinstance(checkpoint, list) or isinstance(checkpoint, tuple)
        for i, learner in self.learners.items():
            self.load_agent_checkpoint(checkpoint[i], i, ckpt_dir)
    
    def load_agent_checkpoint(self, ckpt, idx, ckpt_dir=None):
        ckpt_dir = self.model_dir if not ckpt_dir else ckpt_dir

        if isinstance(ckpt, int):
            cp_path = '%s/%s/epoch_%04d.p' % (ckpt_dir, "agent_"+str(idx), ckpt)
        else:
            assert isinstance(ckpt, str)
            cp_path = '%s/%s/%s.p' % (ckpt_dir, "agent_"+str(idx), ckpt)
        model_cp = pickle.load(open(cp_path, "rb"))

        # load model
        self.learners[idx].load_ckpt(model_cp)
        self.logger.info('loading agent_%s model from checkpoint: %s' % (str(idx), cp_path))
        self.logger.info('best reward: %f' % (self.learners[idx].best_reward))
        self.logger.info('epoch: %f' % (self.learners[idx].epoch))

    def save_checkpoint(self, epoch):
        def save(cp_path, idx):
            try:
                pickle.dump(self.learners[idx].save_ckpt(epoch), open(cp_path, 'wb'))
            except FileNotFoundError:
                folder_path = os.path.dirname(cp_path)
                os.makedirs(folder_path, exist_ok=True)
                pickle.dump(self.learners[idx].save_ckpt(epoch), open(cp_path, 'wb'))
            except Exception as e:
                print("An error occurred while saving the model:", e)

        cfg = self.cfg
        additional_saves = cfg.agent_specs.get('additional_saves', None)
        if (cfg.save_model_interval > 0 and (epoch+1) % cfg.save_model_interval == 0) or \
           (additional_saves is not None and (epoch+1) % additional_saves[0] == 0 and epoch+1 <= additional_saves[1]):
            self.writer.flush()
            for i in self.learners:
                save('%s/%s/epoch_%04d.p' % (self.model_dir, "agent_"+str(i), epoch + 1), i)
        for i in self.learners:
            if self.learners[i].save_best_flag:
                self.writer.flush()
                self.logger.critical(f"save best checkpoint with agent_{i}'s rewards {self.learners[i].best_reward:.2f}!")
                save('%s/%s/best.p' % (self.model_dir, "agent_"+str(i)), i)
    
    def display(self, num_episode=3, mean_action=True):
        # total score record: [agent_0_win_times, agent_1_win_times, games_num]
        total_score = [0 for _ in self.learners]
        total_score.append(0)
        total_reward = []
        
        for _ in range(num_episode):
            episode_reward = [0 for _ in self.learners]
            states, info = self.env.reset(symmetric=self.cfg.symmetric)
            # normalize states
            for i, learner in self.learners.items():
                if learner.running_state is not None:
                    states[i] = learner.running_state(states[i])

            for t in range(10000):
                state_var = mix_tensorfy(states)
                use_mean_action = mean_action or torch.bernoulli(torch.tensor([1 - self.noise_rate])).item()
                # select actions
                with torch.no_grad():
                    actions = []
                    for i, learner in self.learners.items():
                        if hasattr(learner, 'flag') and self.env.agents[i].flag == "evo":
                            actions.append(learner.policy_net.select_action([state_var[i]], use_mean_action).squeeze().numpy().astype(np.float64))
                        elif hasattr(learner, 'flag') and self.env.agents[i].flag == "dev":
                            actions.append(learner.policy_net.select_action([state_var[i]], use_mean_action).squeeze().numpy().astype(np.float64))
                        else:
                            actions.append(learner.policy_net.select_action(state_var[i], use_mean_action).squeeze().numpy().astype(np.float64))
                next_states, env_rewards, terminateds, truncated, infos = self.env.step(actions)

                # normalize states
                for i, learner in self.learners.items():
                    if learner.running_state is not None:
                        next_states[i] = learner.running_state(next_states[i])

                # use custom or env reward
                if self.cfg.use_exploration_curriculum:
                    assert hasattr(self, 'custom_reward')
                    c_rewards, c_infos = self.custom_reward(infos)
                    rewards = c_rewards
                else:
                    c_rewards, c_infos = [0.0 for _ in self.learners], [np.array([0.0]) for _ in self.learners]
                    rewards = env_rewards
                
                for i in self.learners:
                    episode_reward[i] += rewards[i]

                if truncated:
                    draw = True
                elif terminateds[0]:
                    for i in self.learners:
                        if "winner" in infos[i]:
                            draw = False
                            total_score[i] += 1
                    masks = [0 for _ in self.learners]
                
                if terminateds[0] or truncated:
                    total_score[-1] += 1
                    break
                states = next_states
            
            total_reward.append(episode_reward)

        def average(list):
            total = sum(list)
            length = len(list)
            return total / length

        self.logger.info("Agent_0 gets averaged episode reward: {:.2f}".format(average(list(zip(*total_reward))[0])))
        self.logger.info("Agent_0 gets win rate over {} rounds: {:.2f}".format(num_episode, total_score[0]/total_score[-1]))

        if self.agent_num > 1:
            self.logger.info("Agent_1 gets averaged episode reward: {:.2f}".format(average(list(zip(*total_reward))[1])))
            self.logger.info("Agent_1 gets win rate over {} rounds: {:.2f}".format(num_episode, total_score[1]/total_score[-1]))
        