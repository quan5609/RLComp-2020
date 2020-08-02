#!/usr/bin/python3

import numpy as np
from MinerEnv import MinerEnv
# from keras.models import model_from_json
import sys
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


ACTION_GO_LEFT = 0
ACTION_GO_RIGHT = 1
ACTION_GO_UP = 2
ACTION_GO_DOWN = 3
ACTION_FREE = 4
ACTION_CRAFT = 5

HOST = "localhost"
PORT = 37313
# if len(sys.argv) == 3:
#     HOST = str(sys.argv[1])
#     PORT = int(sys.argv[2])

# load json and create model
# json_file = open('RLModelSample.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# DQNAgent = model_from_json(loaded_model_json)
# load weights into new model
# DQNAgent.load_weights("RLModelSample.h5")
# print("Loaded model from disk")
status_map = {0: "STATUS_PLAYING", 1: "STATUS_ELIMINATED_WENT_OUT_MAP", 2: "STATUS_ELIMINATED_OUT_OF_ENERGY",
              3: "STATUS_ELIMINATED_INVALID_ACTION", 4: "STATUS_STOP_EMPTY_GOLD", 5: "STATUS_STOP_END_STEP"}
action_map = {0: "GO LEFT", 1: "GO RIGHT",
              2: "GO DOWN", 3: "GO UP", 4: "SLEEP", 5: "DIG GOLD"}
count_step = 0

try:
    # Initialize environment
    minerEnv = MinerEnv(HOST, PORT)
    minerEnv.start()  # Connect to the game
    minerEnv.reset()
    s = minerEnv.get_state()  # Getting an initial state
    while not minerEnv.check_terminate():
        try:
            # Getting an action from the trained model
            # action = np.argmax(DQNAgent.predict(s.reshape(1, len(s))))
            action = minerEnv.get_action()
            print("next action = ", action_map[action])
            # Performing the action in order to obtain the new state
            minerEnv.step(str(action))
            s_next = minerEnv.get_state()  # Getting a new state
            s = s_next
            count_step += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("Finished.")
            break
    minerEnv.end()
    print("After finish, total step: ", count_step)
    print(status_map[minerEnv.state.status])

except Exception as e:
    import traceback
    traceback.print_exc()
print("End game.")
