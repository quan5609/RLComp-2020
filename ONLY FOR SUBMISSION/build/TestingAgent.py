#!/usr/bin/python3
from keras.models import model_from_json
import numpy as np
from MinerEnv import MinerEnv
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

try:
    # Initialize environment
    minerEnv = MinerEnv(HOST, PORT)
    minerEnv.start()  # Connect to the game

    minerEnv.reset()
    print(minerEnv.state.mapInfo.golds)

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
            # current_cluster = clusterId

            agentState = minerEnv.get_agent_state(clusterId)
            action, goldPos = minerEnv.get_action()
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
