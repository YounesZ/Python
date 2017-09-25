"""



 multiarmed_bandit.py  (author: Anson Wong / git: ankonzoid)



 Classical epsilon-greedy agent solving the multi-armed bandit problem.

 Given a set of bandits with a probability distribution of success, we

 maximize our collection of rewards with an agent that explores with

 epsilon probability, and exploits the action of highest value estimate

 for the remaining probability. This experiment is performed many times

 and averaged out and plotted as an averaged reward history.



 The update rule for the values is via an incremental implementation of:



   V(a;k+1) = V(a;k) + alpha*(R(a) - V(a;k))



 where

   k = # of times action "a" (essentially bandit here) was chosen in the past

   V(a;k) = value of action "a"

   R(a) = reward for choosing action (bandit) "a"

   alpha = 1/k



 Note that the reward R(a) is stochastic in this example and follows the probability

 of the distributions provided by the user in the variable "bandits".



"""

import numpy as np

import matplotlib.pyplot as plt



def main():

    # =========================

    # Settings

    # =========================

    bandits = [0.1, 0.4, 0.5, 0.9, 0.1, 0.2, 0.8]  # bandit probabilities of success

    N_experiments = 1000  # number of experiments to perform

    N_pulls = 2000  # number of pulls per experiment

    epsilon = 0.01  # probability of random exploration (fraction)



    save_fig = False  # if false -> plot, if true save as file in same directory



    # =========================

    # Define bandit class

    # =========================

    N_bandits = len(bandits)

    class Bandit:

        def __init__(self, bandits):

            self.prob = bandits  # probabilities of success

            self.n = np.zeros(N_bandits)  # number of times bandit was pulled

            self.V = np.zeros(N_bandits)  # estimated value



        def get_reward(self, action):

            rand = np.random.random()

            if rand < self.prob[action]:

                reward = 1  # success

            else:

                reward = 0  # failure

            return reward



        # choose action based on epsilon-greedy agent

        def choose_action(self, epsilon):

            rand = np.random.random()  # random float in [0.0,1.0)

            if rand < epsilon:

                return np.random.randint(N_bandits)  # explore

            else:

                return np.argmax(self.V)  # exploit



        def update_V(self, action, reward):

            # Update action counter

            self.n[action] += 1

            # Update V via an incremental implementation

            #  V(a;k+1) = V(a;k) + alpha*(R(a) - V(a;k))

            #  alpha = 1/k

            # where

            #  V(a;k) is the value of action a

            #  k is the number of times action was chosen in the past

            alpha = 1./self.n[action]

            self.V[action] += alpha * (reward - self.V[action])



    # =========================

    # Define out experiment procedure

    # =========================

    def experiment(bandit, Npulls, epsilon):

        action_history = []

        reward_history = []

        for i in range(Npulls):

            # Choose action, collect reward, and update value estimates

            action = bandit.choose_action(epsilon)  # choose action (we use epsilon-greedy approach)

            reward = bandit.get_reward(action)  # pick up reward for chosen action

            bandit.update_V(action, reward)  # update our value V estimates for (reward, action)

            # Track action and reward history

            action_history.append(action)

            reward_history.append(reward)

        return (np.array(action_history), np.array(reward_history))



    # =========================

    #

    # Start multi-armed bandit simulation

    #

    # =========================

    print("Running multi-armed bandit simulation with epsilon = {}".format(epsilon))

    reward_history_avg = np.zeros(N_pulls)  # reward history experiment-averaged

    for i in range(N_experiments):

        # Initialize our bandit configuration

        bandit = Bandit(bandits)

        # Perform experiment with epsilon-greedy agent (updates V, and reward history)

        (action_history, reward_history) = experiment(bandit, N_pulls, epsilon)

        if (i+1) % (N_experiments/20) == 0:

            print("[{}/{}] reward history = {}".format(i+1, N_experiments, reward_history))

        # Sum up experiment reward (later to be divided to represent an average)

        reward_history_avg += reward_history



    reward_history_avg /= np.float(N_experiments)

    print("reward history avg = {}".format(reward_history_avg))



    # =========================

    # Plot average reward history results

    # =========================

    plt.plot(reward_history_avg)

    plt.xlabel("iteration #")

    plt.ylabel("average reward (over {} experiments)".format(N_experiments))

    plt.title("Multi-armed bandit with epsilon-greedy agent (epsilon = {})".format(epsilon))

    if save_fig:

        output_file = "multiarmed_bandit_reward_history_avg.pdf"

        plt.savefig(output_file, bbox_inches="tight")

    else:

        plt.show()



# Driver

if __name__ == "__main__":

    main()