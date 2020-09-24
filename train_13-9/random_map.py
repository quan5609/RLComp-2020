import numpy as np
import os
import json
import random
import time

random.seed(int(time.time()))

mapdir = "Maps"
mapdir2 = "Maps2"
map_name = 'map12'
with open(os.path.join(mapdir, map_name), 'r') as f:
    maps= f.read()
map_file = json.loads(maps)

MAP_MAX_X = 21  # Width of the Map
MAP_MAX_Y = 9  # Height of the Map
count_cell = 0

for i in range(MAP_MAX_Y):
    for j in range(MAP_MAX_X):
        if map_file[i][j] == 50:
            count_cell += 1

total_gold = 10000 - count_cell* 50

for i in range(MAP_MAX_Y):
    for j in range(MAP_MAX_X):
        if map_file[i][j] == 50:
            count_cell -= 1
            if count_cell == 0:
                map_file[i][j] += total_gold
                break
            add_gold = random.randint(0, 21) * 50
            add_gold = total_gold if add_gold > total_gold else add_gold
            map_file[i][j] += add_gold
            total_gold -= add_gold
            if total_gold == 0:
                break

with open(os.path.join(mapdir2, map_name), 'w') as outfile:
    json.dump(map_file, outfile)   
