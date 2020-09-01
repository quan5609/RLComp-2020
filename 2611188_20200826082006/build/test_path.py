import os
import numpy as np
import json

for filename in os.listdir(MAP_DIR):
    if filename == "map" + str(MAP_ID):
        with open(os.path.join(MAP_DIR, filename), 'r') as f:
            temp = f.read()
            mymap = json.loads(temp)
            energyOnMap = json.loads(temp)