from MINER_STATE import State
import numpy as np
import sys
from agent_8_8 import Agent_8_8
from agent_11_8 import Agent_11_8
from agent_dummy import Agent_Dummy
import random
from warnings import simplefilter
import time

simplefilter(action='ignore', category=FutureWarning)


TreeID = 1
TrapID = 2
SwampID = 3



class PlayerInfo:
    def __init__(self, id):
        self.playerId = id
        self.score = 0
        self.energy = 0
        self.posx = 0
        self.posy = 0
        self.lastAction = -1
        self.status = 0
        self.freeCount = 0

class BigAgent:
    def __init__(self, agentId, isCLuster = False):
        # self.socket = GameSocket(host, port)

        random.seed(agentId + int(time.time()))
        self.isCluster = isCLuster
        self.agent_id = agentId
        self.info = PlayerInfo(self.agent_id)
        if not self.isCluster:
            self.strategy = [Agent_8_8(agentId), Agent_11_8(agentId), Agent_Dummy(agentId)]
        else:
            self.strategy = [Agent_11_8(agentId)]
        
        
        for agent in self.strategy:
            agent.info = self.info

        self.countStep = 0
        self.currentAgent = 0 if self.isCluster else random.randint(0,2) 
        # self.state = State()
        # self.isSleeping = False
        # self.swampCount = -1
        # self.sleepCount = -1
        # self.sleepBonus = [12, 16, 25]
        # # Storing the last score for designing the reward function
        # self.score_pre = self.state.score

    def reset(self, message):  # start new game
        for agent in self.strategy:
            agent.state.init_state(message)  # init state
            agent.state.id = self.agent_id
    
    def update(self, message):
        for agent in self.strategy:
            agent.state.update_state(message)

    def updateLastAction(self, action):
        self.info.lastAction = action
        for agent in self.strategy:
            agent.info.lastAction = action

    def step(self):  # step process
        if self.countStep % 5 == 0:
            self.currentAgent = 0 if self.isCluster else random.randint(0,2) 
            # print("Debug multi-strategy: ", self.agent_id, self.currentAgent)
        action, goldPos = self.strategy[self.currentAgent].get_action()
        self.countStep += 1
        return action