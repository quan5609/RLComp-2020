#!/usr/bin/python3
from keras.models import model_from_json
import numpy as np
from MinerEnv import MinerEnv
from best_3 import Best_3
from best_2_4 import Best_2_4
import sys
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


# load json and create model
json_file = open('DQNmodel_latest.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
DQNAgent = model_from_json(loaded_model_json)
# load weights into new model
DQNAgent.load_weights("DQNmodel_latest.h5")


ACTION_GO_LEFT = 0
ACTION_GO_RIGHT = 1
ACTION_GO_UP = 2
ACTION_GO_DOWN = 3
ACTION_FREE = 4
ACTION_CRAFT = 5

HOST = "localhost"
# PORT = int(sys.argv[1])
PORT = 1111

if len(sys.argv) == 3:
    HOST = str(sys.argv[1])
    PORT = int(sys.argv[2])

status_map = {0: "STATUS_PLAYING", 1: "STATUS_ELIMINATED_WENT_OUT_MAP", 2: "STATUS_ELIMINATED_OUT_OF_ENERGY",
              3: "STATUS_ELIMINATED_INVALID_ACTION", 4: "STATUS_STOP_EMPTY_GOLD", 5: "STATUS_STOP_END_STEP"}
action_map = {0: "GO LEFT", 1: "GO RIGHT",
              2: "GO UP", 3: "GO DOWN", 4: "SLEEP", 5: "DIG GOLD"}
count_step = 0

prevAction = - 1
prevGoldPos = None
init_pos = [[16, 0], [13, 5], [9, 1], [8, 8], [3, 3]]
init_pos_2_4 = [[13, 5], [8, 8]]
init_pos_3 = [[9, 1]]
# init_pos = [[16, 0], [13, 5], [9, 1], [8, 8], [3, 3]]
agentId = 0

try:
    # Initialize environment
    minerEnv = MinerEnv(HOST, PORT)
    minerEnv.start()  # Connect to the game
    mapID = 1
    # Choosing a initial position of the DQN agent on X-axes randomly
    posID_x = init_pos[mapID-1][0]
    # Choosing a initial position of the DQN agent on Y-axes randomly
    # posID_y = np.random.randint(MAP_MAX_Y)
    posID_y = init_pos[mapID-1][1]
    # Creating a request for initializing a map, initial position, the initial energy, and the maximum number of steps of the DQN agent
    request = ("map" + str(mapID) + "," + str(posID_x) +
               "," + str(posID_y) + ",50,100")
    # print("Debug request:", request)
    # Send the request to the game environment (GAME_SOCKET_DUMMY.py)
    minerEnv.send_map_info(request)
    minerEnv.reset()
    # print(minerEnv.state.mapInfo.golds)
    initial = [minerEnv.state.x, minerEnv.state.y]
    print("Initial:", initial)
    # if initial in init_pos_2_4:
    #     agentId = 1

    # if initial in init_pos_3:
    #     agentId = 2

    if agentId == 0:
        s = minerEnv.get_state()  # Getting an initial state
        while not minerEnv.check_terminate():
            try:
                if minerEnv.check_mining():
                    action, goldPos = minerEnv.get_action()
                    # print("Debug action", action)
                    minerEnv.step(str(action))
                    # prevAction = action
                    # prevGoldPos = goldPos
                    # reward += minerEnv.get_reward()
                    s = minerEnv.get_state()
                    # count_step += 1
                    continue

                # current_state = s
                # print("State:", s)
                clusterId = np.argmax(DQNAgent.predict(s.reshape(1, len(s))))
                if count_step == 0:
                    x, y, _ = minerEnv.getClusterByIndex(None, clusterId)
                    print(clusterId, x, y)
                    print("State:", s)
                    print(DQNAgent.predict(s.reshape(1, len(s))))
                    break
                # current_cluster = clusterId

                agentState = minerEnv.get_agent_state(clusterId, mapID)
                action, goldPos = minerEnv.get_action()
                # print("Debug action", action)
                minerEnv.step(str(action))
                if minerEnv.targetCluster:
                    print("Target:", minerEnv.targetCluster._id,
                          minerEnv.targetDesx, minerEnv.targetDesy, minerEnv.state.x, minerEnv.state.y)
                print("Action:", action)
                s = minerEnv.get_state()  # Getting a new state
                ''' Get reward '''
                # reward += minerEnv.get_reward()  # Getting a reward
                count_step += 1
            except Exception as e:
                import traceback
                traceback.print_exc()
                print("Finished.")
                break
        minerEnv.end()
        print("After finish:\n\tEnd status: %s\n\tTotal step: %d" %
              (status_map[minerEnv.state.status], count_step))

    elif agentId == 1:
        print("BEST 2 4")
        virtual_agent = Best_2_4(state=minerEnv.state)
        while not minerEnv.check_terminate():
            try:
                print(
                    "#################################################################")

                action, goldPos = virtual_agent.get_action()
                if(action == 0 and prevAction == 1) or (action == 1 and prevAction == 0) or (action == 2 and prevAction == 3) or (action == 3 and prevAction == 2):
                    print("Prev gold: ", prevGoldPos)
                    print("current gold: ", goldPos)

                print("At step %d:\n\tCurrent gold: %d\n\tCurrent energy: %d\n\tAction: %s" % (
                    count_step, minerEnv.state.score, minerEnv.state.energy, action_map[action]))
                minerEnv.step(str(action))
                prevAction = action
                prevGoldPos = goldPos
                # s_next = minerEnv.get_state()  # Getting a new state
                # s = s_next
                count_step += 1
                # if count_step > 33:
                #     break
            except Exception as e:
                import traceback
                traceback.print_exc()
                print("Finished.")
                break
        minerEnv.end()
        print("After finish:\n\tEnd status: %s\n\tTotal step: %d" %
              (status_map[minerEnv.state.status], count_step))
    # else:
    #     while not minerEnv.check_terminate():
    #         try:
    #             print(
    #                 "#################################################################")
    #             print("BEST 3")
    #             virtual_agent = Best_3(state=minerEnv.state)
    #             action, goldPos = virtual_agent.get_action()
    #             if(action == 0 and prevAction == 1) or (action == 1 and prevAction == 0) or (action == 2 and prevAction == 3) or (action == 3 and prevAction == 2):
    #                 print("Prev gold: ", prevGoldPos)
    #                 print("current gold: ", goldPos)

    #             print("At step %d:\n\tCurrent gold: %d\n\tCurrent energy: %d\n\tAction: %s" % (
    #                 count_step, minerEnv.state.score, minerEnv.state.energy, action_map[action]))
    #             minerEnv.step(str(action))
    #             prevAction = action
    #             prevGoldPos = goldPos
    #             # s_next = minerEnv.get_state()  # Getting a new state
    #             # s = s_next
    #             count_step += 1
    #             # if count_step > 33:
    #             #     break
    #         except Exception as e:
    #             import traceback
    #             traceback.print_exc()
    #             print("Finished.")
    #             break
    #     minerEnv.end()
    #     print("After finish:\n\tEnd status: %s\n\tTotal step: %d" %
    #           (status_map[minerEnv.state.status], count_step))

except Exception as e:
    import traceback
    traceback.print_exc()
print("End game.")
