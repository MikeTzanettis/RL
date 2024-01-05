import numpy as np
import random
from kubernetes import client, config
from prometheus_api_client import PrometheusConnect
import time
import requests
import subprocess
import os
class Environment:
    def __init__(self,prometheus_url):
        self.deployment_name = "deploy-warehouseapi"
        self._prometheus_url = prometheus_url
        self.under_utilization_threshold = 2
        self.low_utilization_threshold = 10
        self.normal_utilization_threshold = 15
        self.over_utilization_threshold = 25
        self.max_replica_count = 6
        self.min_replica_count = 1
        self.current_state = self.get_state()
        self.is_terminal_state = False

    def reset(self):
        replica_count = random.randint(self.min_replica_count,self.max_replica_count)

        k6_script_path = "/home/miketz/Desktop/Service1/script.js"

        # Build the command to run k6 with your script
        command = ["k6", "run","-e", f"RATE={500}","-e", f"PRE_ALLOCATED_VUS={100}", k6_script_path]

        # Run the k6 script from Python
        try:
            subprocess.Popen(command)
        except subprocess.CalledProcessError as e:
            print(f"Error running k6: {e}")
        """
            Reset Environment to a random non terminal state (10,20,30,40)
            at the start of the episode
            by setting appropriate workload
        """

    def render(self):
        """
        render useful information
        """

    def get_state(self):
        print("Getting average cpu...")
        try:
            average_cpu = self.get_avg_irate_cpu_percentage()
            number_of_pods = self.get_current_replica_count()
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Prometheus: {e}")
            return None

        print(f"CPU utilization is at {average_cpu} with {number_of_pods} pods.")
       
        if average_cpu > self.over_utilization_threshold:
            print("OVER UTILIZATION")
            cpu_state = 4
        elif self.normal_utilization_threshold <= average_cpu <= self.over_utilization_threshold:
            print("HIGH UTILIZATION")
            cpu_state = 3
        elif self.low_utilization_threshold <= average_cpu <= self.normal_utilization_threshold:
            print("NORMAL UTILIZATION")
            cpu_state = 2
        elif self.under_utilization_threshold <= average_cpu <= self.low_utilization_threshold:
            print("LOW UTILIZATION")
            cpu_state = 1
        else:
            print("UNDER UTILIZATION")
            cpu_state = 0

        return cpu_state,number_of_pods
    
    def get_avg_irate_cpu_percentage(self):
        # Connect to Prometheus
        prom = PrometheusConnect(url=self._prometheus_url, disable_ssl=True)

        # Prometheus query
        query = 'avg(100 * irate(process_cpu_seconds_total{job="node-warehouseapi", service="service-warehouseapi"}[1m]))'

        # Execute query
        result = prom.custom_query(query)
        average_cpu = result[0]["value"][1]
        return round(float(average_cpu), 2)

    def is_valid_action(self,action):
        current_replica_count = self.get_current_replica_count()
        threshold = current_replica_count + action
        print(f"----is_valid_action---- threshold: {threshold}")
        print(f"Current replica count: {current_replica_count}")
        if threshold > self.max_replica_count or threshold < self.min_replica_count:
            return False
        return True

    def get_current_replica_count(self,namespace="default"):
        config.load_kube_config()  # Load kubeconfig file for local testing; use config.load_incluster_config() for in-cluster usage

        apps_v1 = client.AppsV1Api()
        deployment = apps_v1.read_namespaced_deployment(self.deployment_name, namespace)
        #print(deployment)
        return deployment.spec.replicas

    def scale_deployment(self, scale_count, namespace="default"):
        config.load_kube_config()  # Load kubeconfig file for local testing; use config.load_incluster_config() for in-cluster usage

        apps_v1 = client.AppsV1Api()
        deployment = apps_v1.read_namespaced_deployment(self.deployment_name, namespace)
        deployment.spec.replicas = scale_count
        apps_v1.patch_namespaced_deployment(
            name=self.deployment_name,
            namespace=namespace,
            body=deployment
        )

    def perform_action(self,action):
        current_replica_count = self.current_state[1]
        scale_count = current_replica_count + action
        new_state = self.current_state
        if scale_count == 0:
            print("Invalid Scaling decesion! Reward: -100")
            reward = -100
            is_state_terminal = True
            return reward, self.current_state,is_state_terminal

        new_state = self.current_state
        if(action != 0):
            print(f"Action taken: {action} replicas")
            self.scale_deployment(scale_count)

            # Wait for action to take effect
            time.sleep(30)
            print(f"Scaled to {scale_count} replicas from {current_replica_count} replicas")
        else:
            print(f"No action taken: {action}")

        new_state = self.get_state()
        reward,is_state_terminal = self.calculate_reward(new_state)
        self.current_state = new_state

        return reward, new_state,is_state_terminal

    def calculate_reward(self,current_state):
        cpu_state, number_of_pods = current_state
        reward = 0
        is_state_terminal = False
        if (cpu_state == 0 and number_of_pods != 1) or cpu_state == 4:
            reward = -number_of_pods
            is_state_terminal = True
        elif cpu_state == 1 or cpu_state == 3:
            reward = -number_of_pods / 2
        else:
            reward = 10 - number_of_pods
        
        print(f"Reward is: {reward}")
        return reward,is_state_terminal

    def step(self,action):
        # Apply the action to the environment, record the observations
        reward, next_state,is_state_terminal= self.perform_action(action)

        # # Render the grid at each step
        # if self.render_on:
        #     self.render()

        return reward, next_state, is_state_terminal