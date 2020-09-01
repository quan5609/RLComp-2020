#!/usr/bin/python3
from keras.models import model_from_json
import numpy as np
from MinerEnv import MinerEnv
import sys
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


# load json and create model
json_file = open('DQNmodel_latest(1).json', 'r')
loaded_model_json = json_file.read()
json_file.close()
DQNAgent = model_from_json(loaded_model_json)
# load weights into new model
DQNAgent.load_weights("DQNmodel_latest(1).h5")


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

try:
    # Initialize environment
    minerEnv = MinerEnv(HOST, PORT)
    minerEnv.start()  # Connect to the game
    mapID = 5
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

    s = minerEnv.get_state()  # Getting an initial state
    while not minerEnv.check_terminate():
        try:
            # if not minerEnv.targetCluster is not None and minerEnv.agentState.value == 1:
            #     print("NONE TARGET", minerEnv.agentState)
            # if minerEnv.check_mining() or (minerEnv.targetCluster is not None and minerEnv.agentState.value == 1):
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
            # current_cluster = clusterId
            # else:
            # if clusterId >= minerEnv.clusterNum:
            #     print("Chon ngu")
            # if minerEnv.currentCluster is not None:
            #     if minerEnv.sorted_cluster_list[clusterId]._id != minerEnv.currentCluster._id:
            #         print("Change IN", minerEnv.currentCluster._id,
            #               minerEnv.sorted_cluster_list[clusterId]._id)
            # if minerEnv.targetCluster is not None:
            #     if minerEnv.sorted_cluster_list[clusterId]._id != minerEnv.targetCluster._id:
            #         print(
            #             "Change OUT", minerEnv.targetCluster._id, minerEnv.sorted_cluster_list[clusterId]._id)

            agentState = minerEnv.get_agent_state(clusterId)
            # print("")
            if minerEnv.targetCluster is not None:
                print("Target:", minerEnv.targetCluster._id,
                      minerEnv.targetDesx, minerEnv.targetDesy, minerEnv.state.x, minerEnv.state.y)
            if minerEnv.currentCluster is not None:
                print("Current:", minerEnv.currentCluster._id,
                      minerEnv.state.x, minerEnv.state.y, minerEnv.currentCluster.total_gold)
            action, goldPos = minerEnv.get_action()
            print("Action:", action)
            # print("Debug action", action)
            minerEnv.step(str(action))
            s = minerEnv.get_state()  # Getting a new state
            ''' Get reward '''
            # reward += minerEnv.get_reward()  # Getting a reward
            # count_step += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("Finished.")
            break
    minerEnv.end()
    print("After finish:\n\tEnd status: %s\n\tTotal step: %d" %
          (status_map[minerEnv.state.status], count_step))

except Exception as e:
    import traceback
    traceback.print_exc()
print("End game.")
