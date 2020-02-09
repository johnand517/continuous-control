## Project Objectives and Setup

We will use a Deep Deterministic Policy Gradient (DDPG) network to train the agent to solve the problem, 
This approach uses an actor and critic network starting from a randomly initialized set of states, 
and attempts to arrive at an optimal policy for our environment.  To achieve this, the actor network updates the policy
and the evaluation of the action results for the given policy is conducted through maintaining and updating a simultaneous
critic network.

To achieve this, we will require:
- A sequential deep neural network that estimates actions based on the provided state (actor network)
- A sequential deep neural network that estimates Q values given states + actions (critic network)
- For both of the above networks, we will keep a local and target network, in which the local network learns actively on each iteration, and the target network is updated more slowly through a soft update parameter.
- A function that adds noise to our selected actions
- A replay buffer that stores prior experiences as our agent learns from the environment

Our program acts upon the environment by initially choosing actions for the given state more-or-less at random, and then determines the reward, and next state given the action chosen.  This "experience" is then stored in the replay buffer.  We then update the current state to the next state determined from the chosen action and repeat this process.

Learning from the environment happens by choosing a sample of experiences from the replay buffer after a determined number of steps through the environment.  From these sampled experiences, 
we calculate successive actions given the current policy state with experienced next states using our target actor network.  To evaluate, we used the experience rewards and estimated next 
actions to get updated Q values from our target critic network.  We then update our local critic network using these new Q values and the expected Q values from the local critic network.
As part of updating our critic network, we using gradient clipping to reduce the magnitude of the norm of the vector to be less than or equal than 1, to prevent a gradient from getting too large
and causing unexpected divergence.

Training on our actor network then happens by estimating next actions, given our current state, and evaluated against results from the updated local critic network.   


We also include Double DQN as part of our learning process, which stabilizes our learning by calculating our estimated best action from the next state using the local q network, and using those actions to calculate estimated next state reward from our target q network see [ref: Double DQN Paper](https://arxiv.org/abs/1509.06461).

The chosen neural network configuration for both our actor networks is a MLP with two hidden layers (default number of nodes 512 -> 256).
The chosen neural network configuration for both our critic networks is a MLP with three hidden layers (default number of nodes 512 -> 256 -> 128).
All networks use batch normalization on the input parameters.

Hyperparameters used for this approach are provided in the hyperparameters.py file

## Current results

Through applying the above learning agent, we are able to achieve for a single agent a target score (averaged over the prior 100 episodes) of 30 after 122 episodes.  The results of our scores through successive training episodes are as shown:

![Epoch Scores](/common/images/score_by_episode.png "Epoch Scores")

## Areas for improvement

- More exhaustive hyperparameter tuning
- Exploration of training on multiple agents
- Compare and contrast to other policy-based networks. e.g. [A3C](https://arxiv.org/pdf/1602.01783.pdf), [D4PG](https://openreview.net/pdf?id=SyZipzbCb)

