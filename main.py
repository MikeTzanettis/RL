from Environment import Environment
from Agent import Agent
import numpy as np

if __name__ == '__main__':
    episodes = 20
    prometheus_url = "http://192.168.49.2:30002/"
    # Define the number of levels for CPU utilization and the range for deployed pods
    num_cpu_levels = 5
    num_deployed_pods = 8
    num_actions = 3

    env = Environment(prometheus_url=prometheus_url)
    agent = Agent(epsilon=0.9)

    # Initialize Q-values with zeros
    q_values = np.zeros((num_cpu_levels, num_deployed_pods, num_actions))
    actions = [-1,0,1]
    env.reset()
    for episode in range(episodes):
        print(f"\nStarted episode: {episode}")
        env.is_terminal_state = False
        step = 0
        while not env.is_terminal_state:
            print()
            print(f"\nStep {step} of episode {episode}")

            # Select an action using e-greedy algorithm
            previous_state = env.current_state
            action_index = agent.get_action(env.current_state,q_values)
            action = actions[action_index]

            # Agent takes and environment observes the resulting state and calculates
            # the reward based on the observed state
            reward,next_state,is_state_terminal = env.step(action)
            env.is_terminal_state = is_state_terminal
            if(next_state is None):
                continue
            
            # Update q_values using temporal difference
            q_values = agent.learn(q_values,previous_state,next_state,action_index,reward)
            step += 1
        print(q_values)
    print("Training Finished")
    print(q_values)