import gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

# 1. Create the environment
env = gym.make("CartPole-v1")

# 2. Initialize the DQN model
# MlpPolicy means a Multi-Layer Perceptron (fully connected neural network)
model = DQN("MlpPolicy", env, verbose=1, learning_rate=1e-3, buffer_size=10000, learning_starts=1000, batch_size=32, target_update_interval=500)

# 3. Train the model
print("Training the DQN model...")
model.learn(total_timesteps=50000)

# 4. Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")

# 5. Test the trained model
obs = env.reset()
for _ in range(1000):
    env.render()  # Renders the environment visually
    action, _states = model.predict(obs, deterministic=True)  # Predict the next action
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs = env.reset()

env.close()
print("Testing complete.")
