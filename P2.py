import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def poisson_prob(n, lam):
    """
    Calculate the Poisson distribution for lambda lam and n
    :param n: int, # of events to calculate for
    :param lam: int, the expected number of events
    :return: float: the probability of observing n events
    """
    return ((lam ** n) * math.exp(-lam)) / (math.factorial(n))


def poisson_range(lam, threshold=1e-3):
    """
    Finds the range of numbers n for which Poisson(n) is greater than the threshold and their associated probabilities
    :param lam: the expected number of events
    :param threshold: the threshold of what is considered a significant probability
    :return: tuple (int, int) of the range of expected number of events that results in a reasonable probability
    """

    range_num = []
    prob = []

    # start at lam and go backwards to find min
    min_p = lam
    while min_p >= 0 and poisson_prob(min_p, lam) > threshold:
        range_num.insert(0, min_p)
        prob.insert(0, poisson_prob(min_p, lam))
        min_p -= 1

    # start at lam and go forwards to find max
    max_p = lam + 1
    while poisson_prob(max_p, lam) > threshold:
        range_num.append(max_p)
        prob.append(poisson_prob(max_p, lam))
        max_p += 1

    # return range_num, prob
    return list(zip(range_num, prob))


class Environment:
    """
    This class represents the environment (car rental space)
    """

    def __init__(self, version):
        """

        :param version:
        """
        # 0 is original
        # 1 is problem 4.7
        # 2 is extra question (really imbalanced rentals a lot of people like to move cars form A -> B)
        self.version = version

        # 2 locations A and B
        self.EXP_RENT_A = 3  # poisson random variable expected #s
        self.EXP_RENT_B = 4
        self.EXP_RETS_A = 3
        self.EXP_RETS_B = 2

        if self.version == 2:
            self.EXP_RENT_A = 8  # poisson random variable expected #s
            self.EXP_RENT_B = 2
            self.EXP_RETS_A = 3
            self.EXP_RETS_B = 9

        # range of reasonable values to expect (probability > 1e-3)
        self.exp_rent_a_range = poisson_range(self.EXP_RENT_A)
        self.exp_rent_b_range = poisson_range(self.EXP_RENT_B)
        self.exp_rets_a_range = poisson_range(self.EXP_RETS_A)
        self.exp_rets_b_range = poisson_range(self.EXP_RETS_B)



        self.MAX_CARS = 20  # Max # of cars in a location
        self.MAX_FREE_CARS = 10  # Max # of chars that can be parked at a location
        self.MAX_MOVE = 5
        self.MAX_TRUCK_MOVE = 10 # Max move with a truck
        self.MAX_FREE_MOVE = 1 # Free move from co-worker

        self.REWARD_RENT = 10  # Reward for renting car ($)
        self.REWARD_MOVE = -2  # Reward for moving a car
        self.REWARD_PARK = -4  # Reward for having more cars
        self.REWARD_TRUCK = -12 # Reward for using the truck


class Jack:
    """
    The Agent Jack that controls how cars are moved across his 2 locations
    """

    def __init__(self, discount, env):
        self.discount = discount
        self.env = env

        # values[# cars @ A, # cars @ B] for every state
        self.values = np.zeros((self.env.MAX_CARS + 1, self.env.MAX_CARS + 1))

        # polices[# cars @ A, # cars @ B] for every state
        self.polices = np.zeros((self.env.MAX_CARS + 1, self.env.MAX_CARS + 1), dtype=int)

        # List of possible actions that jack can make (# of cars moved from A to B)
        self.actions = range(-env.MAX_MOVE, env.MAX_MOVE + 1)

        if self.env.version == 2: # If using truck more actions possible
            self.actions = range(-env.MAX_TRUCK_MOVE, env.MAX_TRUCK_MOVE + 1)


    def policy_eval(self, tolerance=1e-4, max_iterations=100):
        """

        :param tolerance: what is considered a convergent value function
        :param max_iterations: maximum number of iterations to converge
        :return:
        """
        iteration = 0
        change = 1

        # loop until it converges or too many iterations
        while change > tolerance and iteration < max_iterations:
            iteration += 1
            change = 0

            for a in range(self.values.shape[0]):  # loop through every state
                for b in range(self.values.shape[1]):
                    prev_val = self.values[a][b]  # value at the state
                    self.values[a][b] = self._value_eval(a, b, self.polices[a][b])  # update the value at the state
                    change = max(change, abs(prev_val - self.values[a][b]))  # update change

            print(f"Policy Eval: {iteration} Change:{change}")
        print(f"CONVERGED in {iteration} iterations")

    def _value_eval(self, a, b, action):

        # Calculate how many cars jack moves
        new_a = max(min(a - action, self.env.MAX_CARS), 0)
        new_b = max(min(b + action, self.env.MAX_CARS), 0)

        total_reward = 0

        if self.env.version == 1 and action > 0:
            total_reward = self.env.REWARD_MOVE * abs(action - self.env.MAX_FREE_MOVE)  # accounting the free car
        else:
            total_reward = self.env.REWARD_MOVE * abs(action)

        # Calculating the cost of the 2nd parking lot
        if self.env.version == 1 and new_a > 10:
            total_reward += self.env.REWARD_PARK
        if self.env.version == 1 and new_b > 10:
            total_reward += self.env.REWARD_PARK

        # Calculating the cost of moving with a truck
        if self.env.version == 2 and abs(action) > self.env.MAX_MOVE:
            total_reward += self.env.REWARD_TRUCK

        for rent_a, prob_rent_a in self.env.exp_rent_a_range:
            for rent_b, prob_rent_b in self.env.exp_rent_b_range:
                for rets_a, prob_rets_a in self.env.exp_rets_a_range:
                    for rets_b, prob_rets_b in self.env.exp_rets_b_range:
                        probability = prob_rent_a * prob_rent_b * prob_rets_a * prob_rets_b
                        # Finding how much can actually be rented (can't rent more than there are cars in stock)
                        real_rent_a = min(new_a, rent_a)
                        real_rent_b = min(new_b, rent_b)

                        reward = (real_rent_a + real_rent_b) * self.env.REWARD_RENT  # Calculate renting reward

                        # state after returns and rents
                        temp_new_a = max(min(new_a + rets_a - real_rent_a, self.env.MAX_CARS), 0)
                        temp_new_b = max(min(new_b + rets_b - real_rent_b, self.env.MAX_CARS), 0)

                        # probability = self.env.probabilities[rent_a][rent_b][rets_a][rets_b]
                        total_reward += probability * (reward + self.discount * self.values[temp_new_a][temp_new_b])

        return total_reward

    def policy_imp(self):
        print("Starting policy improvement")
        is_policy_stable = True

        for a in range(self.values.shape[0]):  # loop through every state
            for b in range(self.values.shape[1]):
                old_action = self.polices[a, b]

                max_action = old_action
                max_action_val = self._value_eval(a, b, max_action)
                for action in self.actions:
                    # check if the action will lead to a valid state
                    new_a = a - action
                    new_b = b + action

                    if 0 <= new_a <= self.env.MAX_CARS and 0 <= new_b <= self.env.MAX_CARS:
                        curr_action_val = self._value_eval(a, b, action)
                        if curr_action_val > max_action_val:
                            max_action_val = curr_action_val
                            max_action = action
                self.polices[a, b] = max_action

                if old_action != max_action:
                    is_policy_stable = False

        print(f"Finished policy improvement. Stable? {is_policy_stable}")
        return is_policy_stable

    def export_policy(self, i, prefix):
        ax = sns.heatmap(self.polices, linewidth=0)
        ax.invert_yaxis()

        plt.title(f"Policy for iteration {i}")
        plt.xlabel("#Cars at second location")
        plt.ylabel("#Cars at first location")

        plt.savefig(prefix + '_policy' + str(i) + '.jpeg', dpi=300)
        plt.close()

    def export_value(self, i, prefix):
        ax = sns.heatmap(self.values, linewidth=0)
        ax.invert_yaxis()

        plt.title(f"Values for iteration {i}")
        plt.xlabel("#Cars at second location")
        plt.ylabel("#Cars at first location")

        plt.savefig(prefix + '_value' + str(i) + '.jpeg', dpi=300)
        plt.close()


if __name__ == '__main__':
    environment = Environment(0)
    jack = Jack(0.9, environment)

    is_converged = False
    counter = 0

    policy_arr = []
    value_arr = []

    # jack.save_policy(counter)
    # jack.save_value(counter)

    while not is_converged:
        print(f"Starting training {counter}!")
        counter += 1

        jack.policy_eval(tolerance=1e-2)
        is_converged = jack.policy_imp()

        policy_arr.append(jack.polices)
        value_arr.append(jack.values)

        jack.export_policy(counter, "Test0")
        jack.export_value(counter, "Test0")

    print("Done Training")
    os.system('osascript -e \'display notification "Script completed" with title "Jack Is Done!!"\'')
