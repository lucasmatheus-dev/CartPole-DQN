import random
import tensorflow as tf
import numpy as np
import gym

from model.DQNAgent import DQNAgent
from utils.utils import *
from utils.plots import *

def update_seed(value):
    random.seed(value)
    np.random.seed(value)
    tf.random.set_seed(value)

def change_seed(seed_index):
    seeds = [100, 573, 982, 588, 576, 123, 1337, 1212, 1050, 1989]
    update_seed(seeds[seed_index])

def evaluate_agent(agent, env, num_episodes=20, threshold=495):
    rewards = []
    success_count = 0

    for episode in range(num_episodes):
        reward = evaluate(agent, env)
        rewards.append(reward)

        if reward >= threshold:
            success_count += 1

        print(f"[EVAL] Episódio {episode+1}/{num_episodes} | Reward: {reward}")

    mean_reward = np.mean(rewards)
    return mean_reward, success_count, rewards

def train_loop(agent, env, seed_idx):
    epsilon = 1.0
    total_episodes = 0
    loop = 0

    while True:
        print("------ TRAIN ------")
        agent.epsilon = epsilon

        train_rewards, episodes_taken = train(agent, env, seed_idx, loop)
        loop += 1

        total_episodes += len(train_rewards)
        epsilon = agent.epsilon

        plot(train_rewards, len(train_rewards))

        print("------ EVALUATE ------")
        mean_reward, success_count, eval_rewards = evaluate_agent(agent, env)

        plot(eval_rewards, len(eval_rewards))

        # critério de parada
        if success_count >= 18:
            print(">>> Critério de parada atingido")
            break

    return mean_reward, total_episodes


def run():
    agent_mean_rewards = []
    episodes_taken_list = []

    for seed_idx in range(4):
        print(f"\n=========== SEED {seed_idx} ===========")

        change_seed(seed_idx)

        env = gym.make("CartPole-v1")
        agent = DQNAgent(env)

        mean_reward, total_episodes = train_loop(agent, env, seed_idx)

        agent.save(f"./agents_checkpointsc/cartpole-dqn_{seed_idx}.h5")

        agent_mean_rewards.append(mean_reward)
        episodes_taken_list.append(total_episodes)

    final_plot(agent_mean_rewards, episodes_taken_list)

if __name__ == "__main__":
    run()