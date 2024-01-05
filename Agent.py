import numpy as np
from kubernetes import client, config
import time
class Agent:
    def __init__(self, epsilon=1,discount_factor = 0.9,learning_rate = 0.9):
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
    def get_action(self, current_state,q_values):
        if np.random.random() < self.epsilon:
            print("Choosing action from q-values...")

            # Get the indices with the maximum value in the 1D array at the current state
            max_value_indices = np.where(q_values[current_state] == np.max(q_values[current_state]))[0]

            # Randomly choose one of the indices if there are multiple indices with the same max value
            best_action_index = np.random.choice(max_value_indices)

            return best_action_index
        else:
            print("Choosing random action...")
            return np.random.randint(3)

    def learn(self,q_values,previous_state,next_state,action_index,reward):
        #store the old q value
        old_q_value = q_values[previous_state][action_index]

        # calculate temporal difference
        temporal_difference = reward + (self.discount_factor*np.max(q_values[next_state])) - old_q_value

        #update the q-value for the previous state and action pair
        new_q_value = old_q_value + (self.learning_rate * temporal_difference)
        q_values[next_state][action_index] = new_q_value

        return q_values
