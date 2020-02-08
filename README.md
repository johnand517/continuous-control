# Deep Reinforcement Learning - DDPG Network (Critic / Actor Network) - Unity Continuous Control Project

## Project Environment Description

This project is attempting to solve the problem presented in the [Udacity](https://www.udacity.com/) Deep Reinforcement Learning (DRL) course for Artificial Intelligent that asks to set up a policy based approach to  train the computer to maximize the score provided by the environment.  

The problem uses [Unity's ML Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/) to provide an environment in which there are both yellow and blue bananas on a 2D surface in a 3D space.  The objective is to train a two-jointed robotic arm to maximize the time the end of its appendage stays within a moving target. 

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