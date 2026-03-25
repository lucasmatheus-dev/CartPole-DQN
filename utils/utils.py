import numpy as np

def train(agent, env, index_agent, loop, episodes=180,
          epsilon_min=0.001, epsilon_decay=0.999, train_start=1000):

    state_size = env.observation_space.shape[0]
    reward_list = []

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        done = False
        step = 0
        total_reward = 0.0

        while not done:
            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            # penalidade por falha
            if done and step != env._max_episode_steps - 1:
                reward = -100

            agent.remember(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            step += 1

            # controle de epsilon
            if len(agent.memory) > train_start and agent.epsilon > epsilon_min:
                agent.epsilon *= epsilon_decay

            agent.replay()

        # logging
        with open('./data/results.csv', 'a') as f:
            f.write(f"{index_agent},{loop},{e},{total_reward}\n")

        print(f"ep: {e+1}/{episodes}, reward: {total_reward}, epsilon: {agent.epsilon:.4f}")

        reward_list.append(total_reward)

        # early stop
        if total_reward == 500:
            return reward_list, e

    return reward_list, episodes

def evaluate(agente, env, episodes=1, epsilon=0.001):
      state_size=env.observation_space.shape[0]
      action_size = env.action_space.n

      agente.epsilon = epsilon
      acc_reward = 0.0
      for e in range (episodes):
          state = env.reset()
          state = np.reshape(state, [1, state_size])
          end_flag = False
          i = 0
          while not end_flag:
            #self.amb.render()
            action = agente.act(state)
            prox_state, reward, end_flag, _ = env.step(action)
            state = np.reshape(prox_state, [1,state_size])
            if not end_flag or i == env._max_episode_steps-1:
                reward = reward
            else:
                reward = -100
            acc_reward += reward
            i +=1
      return acc_reward