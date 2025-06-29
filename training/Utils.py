import matplotlib.pyplot as mp

def plot(data, size):
    episodes = []
    for e in range(size):
      episodes.append(e+1)
    if size == 20:
        mp.bar(episodes, data, width=0.5, color='purple')
        mp.title("Test Performance Metrics")
    else:
        mp.plot(episodes,data)
        mp.title("Train Performance Metrics")
    mp.xlabel("Episodes")
    mp.ylabel("Rewards")
    mp.show()

def plotFinal(data_reward, data_episodes):
    agentes = []
    for e in range(len(data_reward)):
        agentes.append(e+1)
    x1 =  np.arange(len(data_reward))
    x2 = [x + 0.25 for x in x1]
    mp.bar(x1, data_reward, width=0.5, label = 'Average Rewards', color='yellow')
    mp.bar(x2, data_episodes, width=0.5, label = 'Average Episodes Taken', color='red')
    mp.xlabel("Agents")
    mp.legend()
    mp.title("Average rewards obtained and episodes taken to complete.")
    mp.show()