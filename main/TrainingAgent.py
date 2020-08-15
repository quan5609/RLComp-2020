#!/usr/bin/python3
import datetime
import pandas as pd
# A class of creating a batch in order to store experiences for the training process
from Memory import Memory
from DQNModel import DQN  # A class of creating a deep q-learning model
import numpy as np
from MinerEnv import MinerEnv
import sys
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


ACTION_GO_LEFT = 0
ACTION_GO_RIGHT = 1
ACTION_GO_UP = 2
ACTION_GO_DOWN = 3
ACTION_FREE = 4
ACTION_CRAFT = 5

MAP_MAX_X = 21  # Width of the Map
MAP_MAX_Y = 9  # Height of the Map

HOST = "localhost"
PORT = 1111

status_map = {0: "STATUS_PLAYING", 1: "STATUS_ELIMINATED_WENT_OUT_MAP", 2: "STATUS_ELIMINATED_OUT_OF_ENERGY",
              3: "STATUS_ELIMINATED_INVALID_ACTION", 4: "STATUS_STOP_EMPTY_GOLD", 5: "STATUS_STOP_END_STEP"}
action_map = {0: "GO LEFT", 1: "GO RIGHT",
              2: "GO UP", 3: "GO DOWN", 4: "SLEEP", 5: "DIG GOLD"}


prevAction = - 1
prevGoldPos = None
reward = 0
current_state = None
current_cluster = None

now = datetime.datetime.now()  # Getting the latest datetime
header = ["Ep", "Step", "Reward", "Total_reward", "Action", "Epsilon",
          "Done", "Termination_Code"]  # Defining header for the save file
filename = "Data/data_" + now.strftime("%Y%m%d-%H%M") + ".csv"
with open(filename, 'w') as f:
    pd.DataFrame(columns=header).to_csv(
        f, encoding='utf-8', index=False, header=True)
'''----------------------------------------------'''
# Parameters for training a DQN model
N_EPISODE = 10000  # The number of episodes for training
MAX_STEP = 1000  # The number of steps for each episode
BATCH_SIZE = 32  # The number of experiences for each replay
MEMORY_SIZE = 100000  # The size of the batch for storing experiences
# After this number of episodes, the DQN model is saved for testing later.
SAVE_NETWORK = 100
# The number of experiences are stored in the memory batch before starting replaying
INITIAL_REPLAY_SIZE = 1000
INPUTNUM = 48  # The number of input values for the DQN model
ACTIONNUM = 8  # The number of actions output from the DQN model
MAP_MAX_X = 21  # Width of the Map
MAP_MAX_Y = 9  # Height of the Map
'''----------------------------------------------'''
DQNAgent = DQN(INPUTNUM, ACTIONNUM)
memory = Memory(MEMORY_SIZE)
train = False
# Start training episodes
for episode_i in range(0, N_EPISODE):
    try:
        count_step = 0
        # Initialize environment
        minerEnv = MinerEnv(HOST, PORT)
        minerEnv.start()  # Connect to the game
        # mapID = 5
        # Choosing a map ID from 5 maps in Maps folder randomly
        mapID = np.random.randint(1, 6)
        # Choosing a initial position of the DQN agent on X-axes randomly
        posID_x = np.random.randint(MAP_MAX_X)
        # Choosing a initial position of the DQN agent on Y-axes randomly
        posID_y = np.random.randint(MAP_MAX_Y)
        # Creating a request for initializing a map, initial position, the initial energy, and the maximum number of steps of the DQN agent
        request = ("map" + str(mapID) + "," + str(posID_x) +
                   "," + str(posID_y) + ",50,100")
        # Send the request to the game environment (GAME_SOCKET_DUMMY.py)
        minerEnv.send_map_info(request)
        minerEnv.reset()
        # print(minerEnv.state.mapInfo.golds)
        total_reward = 0  # The amount of rewards for the entire episode
        terminate = False  # The variable indicates that the episode ends
        # Get the maximum number of steps for each episode in training
        maxStep = minerEnv.state.mapInfo.maxStep
        s = minerEnv.get_state()  # Getting an initial state
        while not minerEnv.check_terminate() and count_step < maxStep:
            # print(
            #     "#################################################################")
            # If agent is mining, not update !
            if minerEnv.check_mining():
                action, goldPos = minerEnv.get_action()
                minerEnv.step(str(action))
                prevAction = action
                prevGoldPos = goldPos
                reward += minerEnv.get_reward()
                s = minerEnv.get_state()
                count_step += 1
                continue

            if current_state is not None:
                # Add this transition to the memory batch
                if len(current_state) < 48 or len(s) < 48:
                    print(s)
                memory.push(current_state, current_cluster,
                            reward, minerEnv.check_terminate(), s)
                # Sample batch memory to train network
                if (memory.length > INITIAL_REPLAY_SIZE):
                    # If there are INITIAL_REPLAY_SIZE experiences in the memory batch
                    # then start replaying
                    # Get a BATCH_SIZE experiences for replaying
                    batch = memory.sample(BATCH_SIZE)
                    DQNAgent.replay(batch, BATCH_SIZE)  # Do relaying
                    train = True  # Indicate the training starts
                # Plus the reward to the total rewad of the episode
                total_reward = total_reward + reward
                reward = 0
                # s = s_next  # Assign the next state for the next step.

                # Saving data to file
                save_data = np.hstack(
                    [episode_i + 1, count_step + 1, reward, total_reward, current_cluster, DQNAgent.epsilon, minerEnv.check_terminate()]).reshape(1, 7)
                with open(filename, 'a') as f:
                    pd.DataFrame(save_data).to_csv(
                        f, encoding='utf-8', index=False, header=False)

                if minerEnv.check_terminate() == True:
                    # If the episode ends, then go to the next episode
                    break

            current_state = s
            # print("State:", s)
            clusterId = DQNAgent.act(s)
            if clusterId >= minerEnv.clusterNum:
                reward -= 1000
            else:
                if minerEnv.currentCluster is not None:
                    if minerEnv.sorted_cluster_list[clusterId]._id != minerEnv.currentCluster._id:
                        reward -= 100
                if minerEnv.targetCluster is not None:
                    if minerEnv.sorted_cluster_list[clusterId]._id != minerEnv.targetCluster._id:
                        reward -= 100
            current_cluster = clusterId

            agentState = minerEnv.get_agent_state(clusterId)
            action, goldPos = minerEnv.get_action()
            # print("At step %d:\n\tCurrent gold: %d\n\tCurrent energy: %d\n\tAction: %s" % (
            #     count_step, minerEnv.state.score, minerEnv.state.energy, action_map[action]))
            minerEnv.step(str(action))
            prevAction = action
            prevGoldPos = goldPos
            s = minerEnv.get_state()  # Getting a new state
            ''' Get reward '''
            reward += minerEnv.get_reward()  # Getting a reward
            count_step += 1
            # Iteration to save the network architecture and weights
        if (np.mod(episode_i + 1, SAVE_NETWORK) == 0 and train == True):
            # Replace the learning weights for target model with soft replacement
            DQNAgent.target_train()
            # Save the DQN model
            now = datetime.datetime.now()  # Get the latest datetime
            DQNAgent.save_model("TrainedModels/",
                                "DQNmodel_" + now.strftime("%Y%m%d-%H%M") + "_ep" + str(episode_i + 1))

        # Print the training information after the episode

        print('Episode %d ends. Number of steps is: %d. Accumulated Reward = %.2f. Epsilon = %.2f .Termination code: %d' % (
            episode_i + 1, count_step + 1, total_reward, DQNAgent.epsilon, minerEnv.check_terminate()))
        # Decreasing the epsilon if the replay starts
        if train == True:
            DQNAgent.update_epsilon()
            count_step += 1
            # if count_step > 33:
            #     break
        minerEnv.end()
        print("After finish map %d:\n\tEnd status: %s\n\tTotal step: %d\n\tTotal score: %d" %
              (mapID, status_map[minerEnv.state.status], count_step, minerEnv.state.score))

    except Exception as e:
        import traceback
        traceback.print_exc()
    print("End game.")
