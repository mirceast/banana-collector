# Report

In this project, we trained agents to solve the task of collecting as many yellow bananas as possible in 300 time steps, while avoiding blue bananas. The environment is implemented in [unityagents](https://pypi.org/project/unityagents/#description).

Each state is 37-dimensional and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.

### Implementation
We use the [DQN method](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf) to train our reinforcement learning agents. By interacting with the environment and receiving rewards, the agent improves the estimate of the state-action value function Q(s,a). If this estimate converges to the true state-action value function, the optimal policy is easily recovered by always choosing the action with the largest expected return (greedy action selection). 

The following hyper-parameters were used in the Q learning algorithm:

Discount factor gamma = 0.99           
Soft update weight tau = 1e-3            
Learning rate = 5e-4       
Initial epsilon for selecting the greedy action = 1.0
Minimum epsilon = 0.01
Epsilon decay = 0.995

Performance was good enough so no hyper-parameter search was needed. Besides, I found that it's more interesting to implement an improvement to the DQN method and evaluate its performance instead of fine-tuning parameters. 

Q learning traditionally uses a table to store the state-action value function. In our continuous case, a feed-forward neural network was used to approximate Q(s,a). The architecture consists of three layers:
- 37 neurons in the input layer 
- 64 neurons in the first hidden layer, followed by ReLU activation functions
- 64 neurons in the second hidden layer, also followed by ReLU
- 4 neurons in the output layer with no activation function

During learning, we used a batch size of 64. The neural network learns to estimate the value of each of the four possible actions given the state as an input. The action with the largest associated Q value is then selected with (1 - epsilon) probability. 

It is a bit funny to call it **deep** Q learning when our neural net architecture is in fact so small. I stuck with the DQN name because of the other two major improvements the original paper brought. The first is a replay buffer from which to sample the agent's experiences during learning; this breaks apart the temporal correlations between consecutive states. The second improvement is the use of a separate "target" network, with weights that are updated less often than those of the "online" network. This breaks the correlations between the target Q values and the network's output. We used a replay buffer size of 1e5 and updated the target network every 4 updates of the online network. 

### Double DQN
Beyond training a single agent to solve the task, we extend the goal to a comparison between [vanilla DQN](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf) and [double DQN](https://arxiv.org/abs/1509.06461). Double DQN improves on the traditional version by decoupling action selection and action evaluation. Double DQN proposes to use the online network for choosing the next action, and evaluate that choice with the target network. This was shown to reduce the overoptimism of Q learning and to provide usually better performance.

I ran each algorithm 10 times and compared the average performance. Since each run of 600 episodes took about 350s, it was useful to save each result to disk and simply load it when running the notebook again. I also saved the 10 random seeds to a file and re-used them since we want to have the same seeds when comparing the two algorithms and we don't want them re-generated when running the notebook again. 

### Performance analysis
Let's take a look at the agent's performance in random repetitions. We will also show the *trend* of the data by smoothing it with a window 100 episodes long that's *centered* on the current data point (episode). Note that this is different from the requirement to solve the environment - in that case the current episode is the *last* data point in the sliding window. 

<img src="Single-run performance.png">

Above, the vertical gray bar is the episode at which the environment was solved. Running the cell multiple times suggests that double DQN does solve the environment faster.

Let's see the mean and standard deviation of the two algorithms for all 10 repetitions and get a more general view on performance. We will also show on average, at which episode was the environment solved, for each algorithm. 

<img src="Average performance.png">

We can see double DQN performing a bit better on average than the vanilla DQN. The difference does not seem significant though, as the performance of double DQN is less consistent - see the larger standard deviation in the bar plot. That said, there does seem to be a trend toward faster learning of double DQN starting with episode 350 onward. And let us remember that in one repetition, vanilla DQN didn't even solve the environment in the allocated 600 episodes!

While I expected double DQN to provide vastly better performance, it is worth noting that in the [double DQN paper](https://arxiv.org/abs/1509.06461) there were a few Atari games in which vanilla DQN actually performed better, for instance Chopper Command, Time Pilot, Robotank, and Assault. I think that a more complex task would widen the performance gap. 

### Potential improvements
Double DQN has already proven itself a potential improvement for our banana-collecting agent. I've only looked at double DQN because it's extremely simple to implement - it's just another line of code. Other improvements might be:
- [Prioritized experience replay](https://arxiv.org/abs/1511.05952). This one I think would improve performance dramatically. The idea behind it is very simple: instead of uniformly sampling from the experience replay buffer, why not prioritize experiences from which the agent learned more? The easy way is to achieve it is to scale the probabilities by the TD error. 
- [Dueling DQN](https://arxiv.org/abs/1511.06581). Dueling DQN architectures decouple the values of states and actions. Certain states are better than others without even considering the possible action choices. The dueling architecture allows the network to take advantage of this.
- Hyperparameter search. This is almost guaranteed to improve results but it's time consuming and perhaps not as interesting as implementing various proposed improvements. 