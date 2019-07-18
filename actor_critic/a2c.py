# Forked from https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On
# 7-14-2019 Added documentation while learning the code

import gym

# PyTorch Agent Net library (PTAN) consists of multiple helper functions to interface with the OpenAI environments 
# from pre-processing to generating episodes and playing out policies.
import ptan    

import numpy as np
import argparse
from tensorboardX import SummaryWriter   # tensorboard for pytorch

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim

from lib import common

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01   # Use to induce explore
BATCH_SIZE = 128
NUM_ENVS = 50   # num of env vs batch size ???

BELLMAN_STEPS = 4
CLIP_GRAD = 0.1
REWARD_GOAL = 19.5

if __name__ == "__main__":
    common.mkdir('.', 'checkpoints')
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    # make a batch of env
    make_env = lambda: ptan.common.wrappers.wrap_dqn(gym.make("PongNoFrameskip-v4"))
    envs = [make_env() for _ in range(NUM_ENVS)]

    writer = SummaryWriter(comment="-pong-a2c_" + args.name)  # log data for consumption and visualization by TensorBoard

    # Create A2C policy network
    net = common.AtariA2C(envs[0].observation_space.shape, envs[0].action_space.n).to(device)
    print(net)

    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True, device=device)  # assign A2C to agent
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=BELLMAN_STEPS)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    batch = []

    with common.RewardTracker(writer, stop_reward=REWARD_GOAL) as tracker:   # Run until reward goal reached
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:   # ??
            for step_idx, exp in enumerate(exp_source):    # ??
                batch.append(exp)   # ??

                # handle new rewards
                new_rewards = exp_source.pop_total_rewards()  # ??
                if new_rewards:
                    finished, save_checkpoint = tracker.reward(new_rewards[0], step_idx)
                    if save_checkpoint: 
                        torch.save(net.state_dict(), './checkpoints/' + args.name + "-best.dat")  # save newest "best" model
                    if finished:
                        break

                if len(batch) < BATCH_SIZE:
                    continue

                # When a full batch, perform a policy update
                states_v, actions_t, q_vals_v = common.unpack_batch(batch, net, last_val_gamma=GAMMA**BELLMAN_STEPS, device=device)
                batch.clear()

                optimizer.zero_grad()
                logits_v, value_v = net(states_v)
                
                loss_value_v = F.mse_loss(value_v.squeeze(-1), q_vals_v)

                log_prob_v = F.log_softmax(logits_v, dim=1)
                adv_v = q_vals_v - value_v.detach()   # calculate advantage = Q(s,a) - V(s)
                log_prob_actions_v = adv_v * log_prob_v[range(BATCH_SIZE), actions_t]
                loss_policy_v = -log_prob_actions_v.mean()

                prob_v = F.softmax(logits_v, dim=1)
                entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()   # Entropy - for exploration

                # calculate policy gradients only
                loss_policy_v.backward(retain_graph=True)
                grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                        for p in net.parameters()
                                        if p.grad is not None])

                # apply entropy and value gradients
                loss_v = entropy_loss_v + loss_value_v
                loss_v.backward()
                nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                optimizer.step()
                # get full loss
                loss_v += loss_policy_v

                # Write to tensorboard output file
                tb_tracker.track("advantage",       adv_v, step_idx)
                tb_tracker.track("values",          value_v, step_idx)
                tb_tracker.track("batch_rewards",   q_vals_v, step_idx)
                tb_tracker.track("loss_entropy",    entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy",     loss_policy_v, step_idx)
                tb_tracker.track("loss_value",      loss_value_v, step_idx)
                tb_tracker.track("loss_total",      loss_v, step_idx)
                tb_tracker.track("grad_l2",         np.sqrt(np.mean(np.square(grads))), step_idx)
                tb_tracker.track("grad_max",        np.max(np.abs(grads)), step_idx)
                tb_tracker.track("grad_var",        np.var(grads), step_idx)
