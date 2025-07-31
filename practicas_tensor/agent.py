from tensor import Enviroment,Agent
import sys,time,os
import matplotlib.pyplot as plt

environ=Enviroment(sys.argv[1])
agent=Agent(environ.get_state().shape)

episode_rewards=[]

plt.ion()

done=False

for episode in range(1000):
    state=environ.reset()
    state=environ.get_state()

    rewards=0

    for step in range(80):

        action=agent.get_action(state)

        reward,next_state,done=environ.step(action)

        environ.render()
        time.sleep(0.3)
        os.system("clear")

        next_state=environ.get_state()

        agent.remember(state,action,reward,next_state,done)
        agent.replay()

        state=next_state
        rewards+=reward

        if done:
            break
    episode_rewards.append(rewards)

    if agent.epsilon > agent.epsilon_min:
        agent.epsilon*= agent.epsilon_decay

    if done:
        break

    print("Episodio:", episode, "| Recompensa total:", rewards, "| Epsilon:", agent.epsilon)

plt.plot(episode_rewards)
plt.xlabel("Episodio")
plt.ylabel("Recompensa Total")
plt.title("Recompensas")
plt.grid(True)
plt.savefig("image.png")
