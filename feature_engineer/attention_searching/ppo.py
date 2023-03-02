from .model_rl import Actor
import torch
import torch.optim as optim
import os
from torch.distributions.categorical import Categorical
import logging
import numpy as np


class PPO(object):
    def __init__(self, args, data_nums, operations_c, operations_d, d_model, d_k, d_v, d_ff, n_heads,
                 device, dropout=None, c_param=False):
        self.args = args
        self.entropy_weight = args.entropy_weight

        self.epochs = args.epochs
        self.episodes = args.episodes
        self.ppo_epochs = args.ppo_epochs
        self.operations_c = operations_c

        self.device = device

        if operations_c:
            self.actor_c = Actor(args, data_nums, operations_c, d_model, d_k, d_v, d_ff, n_heads, dropout=dropout, enc_load_pth=args.enc_c_pth).to(self.device)
            self.actor_c_opt = optim.Adam(params=self.actor_c.parameters(), lr=args.lr)
        if operations_d:
            self.actor_d = Actor(args, data_nums, operations_d, d_model, d_k, d_v, d_ff, n_heads, dropout=dropout, enc_load_pth=args.enc_d_pth).to(self.device)
            self.actor_d_opt = optim.Adam(params=self.actor_d.parameters(), lr=args.lr)
        if c_param:
            self.actor_c.load_state_dict(torch.load(c_param)["net"])
            self.actor_c_opt.load_state_dict(torch.load(c_param)["opt"])

        self.baseline = {}
        for step in range(args.steps_num):
            self.baseline[step] = None

        self.baseline_weight = self.args.baseline_weight

        self.clip_epsion = 0.2

    def choose_action_c(self, input_c, step, epoch, ops, sample_rule):
        actions = []
        log_probs = []

        self.actor_c.train()
        action_softmax, m1_output, m2_output, m3_output = self.actor_c(input_c.to(self.device), step)

        index_none = []
        for index, out in enumerate(action_softmax):
            dist = Categorical(out)
            if index in index_none:
                action = torch.tensor(len(ops) - 1).to(self.device)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            actions.append(int(action.item()))
            log_probs.append(log_prob.item())
        return actions, log_probs, m1_output, m2_output, m3_output, action_softmax

    def choose_action_d(self, input_c, step, epoch, ops, sample_rule):
        actions = []
        log_probs = []

        self.actor_d.train()
        action_softmax, m1_output, m2_output, m3_output = self.actor_d(input_c.to(self.device), step)
        index_none = []
        for index, out in enumerate(action_softmax):
            dist = Categorical(out)
            if index in index_none:
                action = torch.tensor(len(ops) - 1).to(self.device)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            actions.append(int(action.item()))
            log_probs.append(log_prob.item())
        return actions, log_probs, m1_output, m2_output, m3_output, action_softmax

    def predict_action_c(self, input_c, step):
        actions = []
        outs = self.actor_c(input_c.to(self.device), step)
        if step == "selector_c":
            for out in outs:
                if out > 0.5:
                    action = 1
                else:
                    action = 0
                actions.append(action)
        else:
            for out in outs:
                action = out.argmax()
                actions.append(action)
        return actions

    def update_c(self, workers_c):
        rewards = []
        dones = []
        for worker in workers_c:
            rewards.extend(worker.accs)
            dones.extend(worker.dones)

        rewards_convert = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.args.gama * discounted_reward)
            rewards_convert.insert(0, discounted_reward)
        for step in range(self.args.steps_num):
            for reward in rewards_convert[step::self.args.steps_num]:
                if self.baseline[step] == None:
                    self.baseline[step] = reward
                else:
                    self.baseline[step] = self.baseline[step] * self.baseline_weight + reward * (
                            1 - self.baseline_weight)
        baseline_step = []
        for step in range(self.args.steps_num):
            baseline_step.append(self.baseline[step])

        baseline_step = torch.tensor(baseline_step, device=self.device)
        self.baseline_step = baseline_step
        rewards_convert = torch.tensor(rewards_convert, device=self.device).reshape(-1, self.args.steps_num)
        advantages = rewards_convert - baseline_step
        # times = mean_after / mean_before

        # Move tensor in worker to self.device
        for worker_index, worker in enumerate(workers_c):
            for i, x in enumerate(worker.log_probs):
                for j, item in enumerate(x):
                    workers_c[worker_index].log_probs[i][j] = torch.tensor(item, device=self.device)
            for i, x in enumerate(worker.actions):
                for j, item in enumerate(x):
                    workers_c[worker_index].actions[i][j] = torch.tensor(item, device=self.device)
            for index, state in enumerate(worker.states):
                workers_c[worker_index].states[index] = state.to(self.device)

        for epoch in range(self.args.ppo_epochs):
            total_loss = 0
            total_loss_actor = 0
            total_loss_entorpy = 0
            for worker_index, worker in enumerate(workers_c):
                old_log_probs_ = worker.log_probs
                states = worker.states
                actions = worker.actions
                steps = worker.steps

                advantage = advantages[worker_index]
                advantage_convert = []

                for i, log_pros in enumerate(old_log_probs_):
                    advantage_ = advantage[i]
                    for j, log_pro in enumerate(log_pros):
                        advantage_convert.append(advantage_)
                advantage_convert = torch.tensor(advantage_convert, device=self.device)

                old_log_probs = torch.tensor([item for x in old_log_probs_ for item in x], device=self.device)

                new_log_probs = []
                entropys = []
                for index, state in enumerate(states):
                    action = actions[index]
                    step = steps[index]
                    action_softmax, m1_output, m2_output, m3_output = self.actor_c(state.to(self.device), step)
                    for k, out in enumerate(action_softmax):
                        dist = Categorical(out)
                        entropy = dist.entropy()
                        entropys.append(entropy.unsqueeze(dim=0))
                        new_log_prob = dist.log_prob(action[k]).unsqueeze(dim=0).float()
                        new_log_probs.append(new_log_prob)

                new_log_probs = torch.cat(new_log_probs)
                entropys = torch.cat(entropys)

                # ppo
                prob_ratio = new_log_probs.exp() / old_log_probs.exp()
                weighted_probs = advantage_convert * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.clip_epsion,
                                                     1 + self.clip_epsion) * advantage_convert
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs)
                actor_loss = actor_loss.sum()

                entropy_loss = entropys.sum()
                total_loss_actor += actor_loss
                total_loss_entorpy += (- self.args.entropy_weight * entropy_loss)
                total_loss += (actor_loss - self.args.entropy_weight * entropy_loss)
            factor = len(workers_c)
            total_loss /= factor
            actor_loss = total_loss_actor / factor
            entropy_loss = total_loss_entorpy / factor
            logging.info(
                f"total_loss_c:{total_loss.item()},actor_loss:{actor_loss.item()},entory_loss:{entropy_loss.item()}")
            self.actor_c_opt.zero_grad()
            total_loss.backward()
            self.actor_c_opt.step()

    def update_d(self, workers_d):
        rewards = []
        dones = []
        for worker in workers_d:
            rewards.extend(worker.accs)
            dones.extend(worker.dones)

        rewards_convert = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.args.gama * discounted_reward)
            rewards_convert.insert(0, discounted_reward)
        if self.operations_c == 0:
            for step in range(self.args.steps_num):
                for reward in rewards_convert[step::self.args.steps_num]:
                    if self.baseline[step] == None:
                        self.baseline[step] = reward
                    else:
                        self.baseline[step] = self.baseline[step] * self.baseline_weight + reward * (
                                1 - self.baseline_weight)
            baseline_step = []
            for step in range(self.args.steps_num):
                baseline_step.append(self.baseline[step])
            baseline_step = torch.tensor(baseline_step, device=self.device)
        else:
            baseline_step = self.baseline_step

        rewards_convert = torch.tensor(rewards_convert, device=self.device).reshape(-1, self.args.steps_num)
        advantages = rewards_convert - baseline_step

        for epoch in range(self.args.ppo_epochs):
            total_loss = 0
            total_loss_actor = 0
            total_loss_entorpy = 0
            for worker_index, worker in enumerate(workers_d):
                old_log_probs_ = worker.log_probs
                states = worker.states
                actions = worker.actions
                steps = worker.steps

                advantage = advantages[worker_index]
                advantage_convert = []

                for i, log_pros in enumerate(old_log_probs_):
                    advantage_ = advantage[i]
                    for j, log_pro in enumerate(log_pros):
                        advantage_convert.append(advantage_)
                advantage_convert = torch.tensor(advantage_convert).to(self.device)

                old_log_probs = torch.tensor([item for x in old_log_probs_ for item in x]).to(self.device)

                new_log_probs = []
                entropys = []
                for index, state in enumerate(states):
                    action = actions[index]
                    step = steps[index]
                    action_softmax, m1_output, m2_output, m3_output = self.actor_d(state.to(self.device), step)
                    if index == 0:
                        softmax_output = action_softmax
                    for k, out in enumerate(action_softmax):
                        dist = Categorical(out)
                        entropy = dist.entropy()
                        entropys.append(entropy.unsqueeze(dim=0))
                        new_log_prob = dist.log_prob(torch.tensor(action[k]).to(self.device)).unsqueeze(dim=0).float()
                        new_log_probs.append(new_log_prob)

                new_log_probs = torch.cat(new_log_probs)
                entropys = torch.cat(entropys)

                # ppo
                prob_ratio = new_log_probs.exp() / old_log_probs.exp()
                weighted_probs = advantage_convert * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.clip_epsion,
                                                     1 + self.clip_epsion) * advantage_convert
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs)
                actor_loss = actor_loss.sum()

                entropy_loss = entropys.sum()
                total_loss_actor += actor_loss
                total_loss_entorpy += (- self.args.entropy_weight * entropy_loss)
                total_loss += (actor_loss - self.args.entropy_weight * entropy_loss)
            total_loss /= len(workers_d)
            actor_loss = total_loss_actor / len(workers_d)
            entropy_loss = total_loss_entorpy / len(workers_d)

            logging.info(
                f"total_loss_d:{total_loss.item()},actor_loss:{actor_loss.item()},entory_loss:{entropy_loss.item()}")
            self.actor_d_opt.zero_grad()
            total_loss.backward()
            self.actor_d_opt.step()

    def save_model_c(self):
        dir = f"./params/dl"
        name = self.args.file_name.split(".")[0]
        if not os.path.exists(dir):
            os.makedirs(dir)
        torch.save({"net": self.actor_c.state_dict(), "opt": self.actor_c_opt.state_dict()},
                   f"{dir}/{name}_actor_c.pth")

