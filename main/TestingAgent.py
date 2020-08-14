#!/usr/bin/python3
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

    # s = minerEnv.get_state()  # Getting an initial state
    while not minerEnv.check_terminate():
        try:
            print("#################################################################")
            action, goldPos = minerEnv.get_action()
            if(action == 0 and prevAction == 1) or (action == 1 and prevAction == 0) or (action == 2 and prevAction == 3) or (action ==3 and prevAction ==2):
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
            if count_step > 33:
                break
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
