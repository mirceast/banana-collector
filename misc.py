from collections import deque
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm


def check_cuda(gpu_idx = 0):
    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    if use_cuda and gpu_idx is not "cpu":
        print('CUDA is available!  Training on GPU ...')
        device = torch.device("cuda:{}".format(gpu_idx))
        print("Using",torch.cuda.get_device_name(device))
    else:
        print('CUDA is not available.  Training on CPU ...')
        device = torch.device("cpu")
    return device


def get_model_save_path(arch, n_episodes, seed):
    return "./temp/" + arch + "_" + "{:03d}".format(n_episodes) + "_eps_" + str(seed) + ".pth"


def get_scores_save_path(arch, n_episodes, seed):
    return "./temp/" + arch + "_" + "{:03d}".format(n_episodes) + "_eps_" + str(seed) + ".np_scores"


def plot_scores(scores, rolling_window=100):
    """Plot scores and optional rolling mean using specified window."""
    plt.plot(np.arange(1, len(scores)+1), scores, alpha=0.5); plt.title("Scores");
    rolling_mean = pd.Series(scores).rolling(rolling_window, center=True).mean()
    plt.plot(rolling_mean);
    plt.show()
    

def dqn(agent, env, solved_score, n_episodes=600, max_t=1000, 
        eps_start=1.0, eps_end=0.01, eps_decay=0.995, verbose=True):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    
    solved_in = None
    
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        while True: # we replace for t in range(max_t) because the max is known, 300 steps
            action = agent.act(state, eps)
            
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        if verbose:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0 and verbose:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= solved_score:
            solved_in = i_episode-100

    return agent, scores, solved_in


def smooth_scores(scores, n_smooth=51):
    if n_smooth%2 == 0:
        raise Exception("Please choose an odd number to use for smoothing")
    n_discarded = int((n_smooth-1)/2)
    smoothed = np.zeros_like(scores)
    sx, sy, sz = scores.shape
    for i in range(sx):
        for j in range(sy):
            smoothed[i,j,:] = pd.Series(scores[i,j,:]).rolling(n_smooth, center=True).mean()

    return smoothed


























