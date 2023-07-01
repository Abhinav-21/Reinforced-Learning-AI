<h1 align='center' style='color:cyan'><b> CAR-RACING-v2 </b></h1>

<h3><b>IMPORTING DEPENDENCIES</b></h3>


```python
import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from gym.wrappers import GrayScaleObservation
import warnings

warnings.filterwarnings("ignore")
```

<h3><b>CREATING ENVIRONMENT</b></h3>


```python
environment_name = "CarRacing-v2"
log_path = os.path.join("Training", "Logs")

# Create the environment

env = gym.make(environment_name, domain_randomize=True, render_mode='human')
env = GrayScaleObservation(env, keep_dim=True)
num_envs = 4  
env = SubprocVecEnv([lambda: gym.make(environment_name, domain_randomize=True) for _ in range(num_envs)])
```

<h3><b>CREATING PPO MODEL, TRAINING IT AND SAVING IT</b></h3>


```python
# Create the PPO model
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

# Train the model
model.learn(total_timesteps=30000)

# Save the trained model
PPO_path = os.path.join("Training", "Saved_Models", "PPO_Model_CarRace")
model.save(PPO_path)
```

<h3><b>LOADING TRAINED MODEL, FOR FURTHER TRAINING</b></h3>


```python
PPO_path = os.path.join("Training", "Saved_Models", "PPO_Model_CarRace")
# Load model
model = PPO.load(PPO_path, env=env)

# Train the model
model.learn(total_timesteps=20000, progress_bar=True)

# Save the trained model
model.save(PPO_path)
```

<h3><b>TESTING THE TRAINED MODEL</b></h3>


```python
# Load the saved model
PPO_path = os.path.join("Training", "Saved_Models", "PPO_Model_CarRace")
model = PPO.load(PPO_path, env=env)

# Create a new environment for evaluation
env = gym.make(environment_name, render_mode='human')
env = env = GrayScaleObservation(env, keep_dim=True)
num_envs = 1
env = SubprocVecEnv([lambda: gym.make(environment_name, domain_randomize=True) for _ in range(num_envs)])

obs = env.reset()
done = False
score = 0 

while not done:
    env.render()
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    score += reward
    
print(f'Score:{score}')

env.close()
```

<h3><b>TESTING THE HEAVILY TRAINED MODEL</b></h3>


```python
environment_name = "CarRacing-v2"
log_path = os.path.join("Training", "Logs")
env = gym.make(environment_name,  render_mode='human')
env = DummyVecEnv([lambda: env])

# Load the saved model
PPO_path = os.path.join("Training", "Saved_Models", "PPO_CarRace_Trained")
model = PPO.load(PPO_path, env=env)

obs = env.reset()
done = False
score = 0 

while not done:
    env.render()
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    score += reward
    
print(f'Score:{score}')

env.close()
```
