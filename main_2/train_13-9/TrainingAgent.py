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
init_pos = [[16, 0], [13, 5], [9, 1], [8, 8], [3, 3]]

now = datetime.datetime.now()  # Getting the latest datetime
# header = ["Ep", "Step", "Reward", "Total_reward", "Action", "Epsilon",
#           "Done", "Termination_Code"]  # Defining header for the save file
# filename = "Data/data_" + now.strftime("%Y%m%d-%H%M") + ".csv"
# with open(filename, 'w') as f:
#     pd.DataFrame(columns=header).to_csv(
#         f, encoding='utf-8', index=False, header=True)
'''----------------------------------------------'''
# Parameters for training a DQN model
N_EPISODE = 10000  # The number of episodes for training
MAX_STEP = 100  # The number of steps for each episode
BATCH_SIZE = 4  # The number of experiences for each replay
MEMORY_SIZE = 100000  # The size of the batch for storing experiences
# After this number of episodes, the DQN model is saved for testing later.
SAVE_NETWORK = 100
# The number of experiences are stored in the memory batch before starting replaying
INITIAL_REPLAY_SIZE = 100
INPUTNUM = 106  # The number of input values for the DQN model
ACTIONNUM = 16  # The number of actions output from the DQN model
MAP_MAX_X = 21  # Width of the Map
MAP_MAX_Y = 9  # Height of the Map


def check_buffer_ready(mem_list):
    for mem in mem_list:
        if mem.length < INITIAL_REPLAY_SIZE:
            return False
    return True


'''----------------------------------------------'''
DQNAgent = DQN(INPUTNUM, ACTIONNUM)
memory_list = []
for mapId in range(1, 13):
    memory_list.append(Memory(mapId, MEMORY_SIZE))
[print(x.mapId) for x in memory_list]
train = False
validLoss = True
# Start training episodes
for episode_i in range(0, N_EPISODE):
    try:
        reward = 0
        current_state = None
        current_cluster = None
        current_cluster_id = None
        if not validLoss:
            break
        episode_memory = []
        count_step = 0
        # Initialize environment
        minerEnv = MinerEnv(HOST, PORT)
        minerEnv.start()  # Connect to the game
        # mapID = 5
        # Choosing a map ID from 5 maps in Maps folder randomly
        mapID = episode_i % 12 + 1
        memory = memory_list[mapID - 1]
        print("BUFFER ID:", memory.mapId)
        # Choosing a initial position of the DQN agent on X-axes randomly
        # posID_x = init_pos[mapID-1][0]
        posID_x = np.random.randint(MAP_MAX_X)
        # Choosing a initial position of the DQN agent on Y-axes randomly
        posID_y = np.random.randint(MAP_MAX_Y)
        # posID_y = init_pos[mapID-1][1]
        # Creating a request for initializing a map, initial position, the initial energy, and the maximum number of steps of the DQN agent
        request = ("map" + str(mapID) + "," + str(posID_x) +
                   "," + str(posID_y) + ",50,100")
        # print("Debug request:", request)
        # Send the request to the game environment (GAME_SOCKET_DUMMY.py)
        minerEnv.send_map_info(request)
        minerEnv.reset()
        # print(minerEnv.state.mapInfo.golds)
        total_reward = 0  # The amount of rewards for the entire episode
        terminate = False  # The variable indicates that the episode ends
        # Get the maximum number of steps for each episode in training
        maxStep = minerEnv.state.mapInfo.maxStep
        s = minerEnv.get_state()  # Getting an initial state
        while not minerEnv.check_terminate() and count_step < maxStep and validLoss:
            # print(
            #     "#################################################################")
            # If agent is mining, not update !
            if minerEnv.check_mining():
                action, goldPos = minerEnv.get_action()
                # print("Debug action", action)
                minerEnv.step(str(action))
                prevAction = action
                prevGoldPos = goldPos
                reward += minerEnv.get_reward()
                s = minerEnv.get_state()
                count_step += 1
                continue

            if current_state is not None and minerEnv.new_decision:
                # Add this transition to the memory batch
                # if len(current_state) < 48 or len(s) < 48:
                #     print(s)
                episode_memory.append(
                    [current_state, current_cluster, reward, minerEnv.check_terminate(), s])
                # Plus the reward to the total rewad of the episode
                minerEnv.new_decision = False
                total_reward = total_reward + reward
                reward = 0

                if minerEnv.check_terminate() == True:
                    # If the episode ends, then go to the next episode
                    break
            current_state = s
            # print("State:", s)
            clusterId = DQNAgent.act(s)

            current_cluster = clusterId

            agentState = minerEnv.get_agent_state(clusterId)
            if minerEnv.new_decision:
                if clusterId >= minerEnv.clusterNum:
                    reward -= 100
                else:
                    desx, desy, _ = minerEnv.getClusterByIndex(
                        minerEnv.state.mapInfo.clusterList, clusterId)
                    distance = abs(minerEnv.state.x - desx) + \
                        abs(minerEnv.state.y-desy)
                    if distance >= 10:
                        reward -= 100
            action, goldPos = minerEnv.get_action()
            # print("At step %d:\n\tCurrent gold: %d\n\tCurrent energy: %d\n\tAction: %s" % (
            #     count_step, minerEnv.state.score, minerEnv.state.energy, action_map[action]))
            # print("Debug action", action)
            minerEnv.step(str(action))
            prevAction = action
            prevGoldPos = goldPos
            s = minerEnv.get_state()  # Getting a new state
            ''' Get reward '''
            # reward += minerEnv.get_reward()  # Getting a reward
            count_step += 1
            # Iteration to save the network architecture and weights

        final_gold = minerEnv.state.score
        print("Final Gold:", final_gold, len(episode_memory))
        total_reward += count_step*(final_gold//50)
        final_gold -= 3200
        # print(final_gold)
        for mem_record in episode_memory:
            current_state, current_cluster, reward, terminate, s = mem_record

            if reward < 0:
                reward -= 64
                reward = np.log(abs(reward)+1)*np.sign(reward)
            else:
                reward += final_gold//50
                reward = (np.log(abs(reward)+1)*np.sign(reward))/50
            # print(reward, np.log(abs(reward)+1)*np.sign(reward))

            memory.push(current_state, current_cluster,
                        reward, terminate, s)
            # Sample batch memory to train network
            if (check_buffer_ready(memory_list)):
                # If there are INITIAL_REPLAY_SIZE experiences in the memory batch
                # then start replaying
                # Get a BATCH_SIZE experiences for replaying
                batch = None
                for mem in memory_list:
                    temp = mem.sample(BATCH_SIZE)
                    # print(temp)
                    if batch is None:
                        batch = temp
                    s, a, r, s2, d = batch
                    curS, curA, curR, curS2, curD = temp
                    s = np.concatenate((s, curS))
                    a = np.concatenate((a, curA))
                    r = np.concatenate((r, curR))
                    s2 = np.concatenate((s2, curS2))
                    d = np.concatenate((d, curD))
                    batch = list([s, a, r, s2, d])

                    # batch += temp
                returnLoss = DQNAgent.replay(
                    batch, BATCH_SIZE*12)  # Do relaying
                if not returnLoss:
                    validLoss = False
                    break
                train = True  # Indicate the training starts

        if (np.mod(episode_i + 1, SAVE_NETWORK) == 0 and train == True):
            # Replace the learning weights for target model with soft replacement
            DQNAgent.target_train()
            # Save the DQN model
            now = datetime.datetime.now()  # Get the latest datetime
            # print("BUGGGGGGGGG")
            DQNAgent.save_model("TrainedModels/",
                                "DQNmodel_latest")
            DQNAgent.save_target_model("TrainedModels/",
                                       "DQNmodel_target_latest")

        # Print the training information after the episode

        print('Episode %d ends. Number of steps is: %d. Accumulated Reward = %.2f. Epsilon = %.2f .Termination code: %d' % (
            episode_i + 1, count_step + 1, total_reward, DQNAgent.epsilon, minerEnv.check_terminate()))
        # Decreasing the epsilon if the replay starts
        if train == True:
            print("Update epsilon: ", episode_i)
            if episode_i % 12 == 0:
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
