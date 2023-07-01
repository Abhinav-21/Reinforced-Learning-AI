# Reinforcement Learning for Game Playing

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A project that demonstrates the use of OpenAI Gym and Stable Baselines3 to train reinforcement learning models to play CartPole and CarRacing games.

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Description

This project showcases the training of reinforcement learning models using the Proximal Policy Optimization (PPO) algorithm from Stable Baselines3. The models are trained to play two games: CartPole and CarRacing, provided by OpenAI Gym.

The goal of the CartPole game is to balance a pole on a cart by moving the cart left or right. The CarRacing game requires driving a car through a racing track, navigating curves and avoiding obstacles.

The project includes code for training the models, saving the trained models, and evaluating their performance. It also provides visualization of the game environment during evaluation.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Abhinav-21/RL-AI.git
   ```
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```
## Usage

#### To use the reinforcement learning game agents, follow the steps below:

1. Open the Jupyter Notebook file `cartpole.ipynb` for training the CartPole agent or `car.ipynb` for training the CarRacing agent.
2. Follow the instructions in the notebook to set up the training environment and configure the hyperparameters.
3. Run the notebook cells to start the training process.
4. Monitor the training progress and observe how the agents improve their gameplay over time.
5. The trained models will be saved in the "Saved_Models" directory.
6. Follow the instructions in the notebook to load the pre-trained models and set up the testing environment.
7. Run the notebook cells to observe the agents' gameplay and evaluate their performance.
8. Analyze the results and compare the agents' performance under different scenarios or configurations.

Feel free to explore and modify the notebooks according to your specific needs. The comments and documentation within the notebooks provide further guidance on the code and usage.

## Contributing

Contributions to this project are welcome! If you have any ideas, suggestions, or bug reports, please feel free to open an issue or submit a pull request. Let's collaborate and improve the game-playing agents together.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
   
