import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

# 1. Create the environment with render mode
env = gym.make("CartPole-v1", render_mode="human")

# 2. Initialize the DQN model with tuned hyperparameters
model = DQN(
    "MlpPolicy", 
    env, 
    verbose=1, 
    learning_rate=1e-3, 
    buffer_size=10000, 
    learning_starts=1000, 
    batch_size=64,          # Increased batch size
    target_update_interval=500, 
    train_freq=4            # Training frequency
)

# 3. Train the model
print("Training the DQN model...")
model.learn(total_timesteps=100000)  # Increased timesteps for better learning

# 4. Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
print(f"Evaluation - Mean Reward: {mean_reward:.2f}, Std Reward: {std_reward:.2f}")

# 5. Save the trained model
model.save("dqn_cartpole_model")
print("Model saved successfully!")

# 6. Load the trained model (just as a demonstration)
loaded_model = DQN.load("dqn_cartpole_model", env=env)
print("Loaded the trained model for testing.")

# 7. Test the trained model visually
obs, _ = env.reset()  # Extract only the observation
for _ in range(1000):
    env.render()  # Render the environment visually
    action, _states = loaded_model.predict(obs, deterministic=True)  # Predict the next action
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()  # Reset the environment

env.close()
print("Testing complete.")
