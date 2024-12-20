# Python AI Gym test 1
import gym
env = gym.make("CartPole-v1")
env.reset()

for _ in range(1000):
    env.render()
    action = env.action_space.sample()  # Take random action
    observation, reward, done, info = env.step(action)
    if done:
        env.reset()

env.close()
