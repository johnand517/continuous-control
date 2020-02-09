# Deep Reinforcement Learning - DDPG Network (Critic / Actor Network) - Unity Continuous Control Project

## Project Environment Description

This project is attempting to solve the problem presented in the [Udacity](https://www.udacity.com/) Deep Reinforcement Learning (DRL) course for Artificial Intelligent that asks to set up a policy based approach to  train the computer to maximize the score provided by the environment.  

The problem contains an environment in which there is a two-jointed robotic, with each joint having two degrees of freedom.  There is a moving target in which we want to maximize the time the end of the robotic arm's appendage stays within the target. the state space is continutous, and the robotic arm has 4 total degrees of freedom, leading to 4 continuous actions.  The objective for this project is to achieve an averaged score of 30 over 100 episodes for a single agent, or to obtain a score of 30 averaged over 20 agents for a single episode.

## Learning Environment Set Up

General instructions are available on the [Udacity github page for this project](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control).

This implementation was validated on a x64 based Windows system using Anaconda3 to provide the hosted environment environment.

1. Setup the conda environment
```
conda create --name continuous-control python=3.6 
activate continuous-control
```
2. Install dependecies
```
# Install general dependencies for Udacity deep-reinforcement-learning projects
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .

# Intall google OpenAI gym dependencies
pip install gym

# Install pytorch 0.4.1
conda install pytorch=0.4.1 cuda92 -c pytorch

# Install the old version of Unity ml-agents
pip install pip install unityagents

# Install the ipython kernel package
conda install ipykernel
# Install the ipython kernel to the conda environment
python -m ipykernel install --user --name continuous-control --display-name "continuous-control"
```

To run the notebook, within the continuous-control conda environment run `jupyter lab` and navigate to the the appropriate ipython notebook.


## Using this program

Learning and a resultant view of both a random state, and learned state are achievable by following the progression in the Navigation-checkpoint.ipynb python notebook.
