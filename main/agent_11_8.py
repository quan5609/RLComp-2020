#!/usr/bin/python3
from MINER_STATE import State
import numpy as np
import enum
import sys
import dbscanner
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


class AgentState(enum.Enum):
    MINING = 0
    GOCLUSTER = 1
    INCLUSTER = 2


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


class Agent_11_8:
    def __init__(self, agentId):
        self.agent_id = agentId
        self.info = PlayerInfo(self.agent_id)
        self.state = State()
        self.isSleeping = False
        self.swampCount = -1
        self.sleepCount = -1
        self.sleepBonus = [12, 16, 25]

        self.agentState = AgentState.GOCLUSTER
        # Storing the last score for designing the reward function
        self.score_pre = self.state.score
        self.currentCluster = None

    def reset(self, message):  # start new game
        self.state.init_state(message)  # init state
        self.state.id = self.agent_id

    def update(self, message):
        self.state.update_state(message)

    def step(self):  # step process
        action, goldPos = self.get_action()
        return action

    def mahattan(self, x1, y1, x2, y2):
        return abs(x1-x2) + abs(y1 - y2)

    def legalAction(self):
        action = [0, 1, 2, 3]
        x, y = self.state.x, self.state.y
        if self.state.x == self.state.mapInfo.max_x or self.state.mapInfo.get_cell_cost(x + 1, y) >= 100:
            action.remove(1)
        if self.state.x == 0 or self.state.mapInfo.get_cell_cost(x - 1, y) >= 100:
            action.remove(0)
        if self.state.y == self.state.mapInfo.max_y or self.state.mapInfo.get_cell_cost(x, y + 1) >= 100:
            action.remove(3)
        if self.state.y == 0 or self.state.mapInfo.get_cell_cost(x, y - 1) >= 100:
            action.remove(2)
        return action

    def get_successor(self, action):
        newX, newY, newEnergy = self.state.x, self.state.y, self.state.energy
        if action == 0:
            newX -= 1
        elif action == 1:
            newX += 1
        elif action == 2:
            newY -= 1
        elif action == 3:
            newY += 1
        deltaE = self.state.mapInfo.get_cell_cost(newX, newY)
        newEnergy -= deltaE
        return newX, newY, newEnergy

    def estimateReceivedGold(self, x, y):
        initGold = self.state.mapInfo.gold_amount(x, y)
        # print("Init gold:", initGold)
        if initGold <= 0:
            return 0
        countPlayer = 0
        for player in self.state.players:
            if player["playerId"] != self.state.id and player["posx"] == x and player["posy"] == y:
                # if player["playerId"] != self.state.id and player["posx"] == x and player["posy"] == y and player["energy"] > 5:
                countPlayer += 1

        if countPlayer == 0:
            return initGold
        if initGold >= (countPlayer + 1) * 50:
            return initGold
        return initGold // (countPlayer + 1)

    def findBestCluster(self):
        bestCluster = None
        globalClusterScore = -1
        bestDestinationx, bestDestinationy = 0, 0
        for cluster in self.state.mapInfo.clusterList:
            if cluster.total_gold > 0:
                estimatePathToCluster, destinationx, destinationy = cluster.distanceToCluster(
                    self.state.x, self.state.y)
                estimateGoldInCluster = cluster.total_gold - 50 * \
                    cluster.checkEnermyInCluster(
                        self.state.players)*(estimatePathToCluster)
                clusterScore = estimateGoldInCluster - 24 * estimatePathToCluster
                clusterScore = max(0, clusterScore)
                if globalClusterScore < clusterScore:
                    globalClusterScore = clusterScore
                    bestDestinationx, bestDestinationy = destinationx, destinationy
                    bestCluster = cluster
        return bestDestinationx, bestDestinationy, bestCluster

    def findBestGoldMine(self):
        pass

    def get_action(self):

        bestValue = -10000
        bestAction = None
        energyOfBest = self.state.energy
        goldPos = None

        actions = self.legalAction()
        goldAmountAtCurrentPosition = self.estimateReceivedGold(
            self.state.x, self.state.y)


        ''' CHANGE STATE '''
        if self.agentState == AgentState.GOCLUSTER:
            desx, desy, bestCluster = self.findBestCluster()
            if desx == self.state.x and desy == self.state.y:
                self.agentState = AgentState.INCLUSTER
                self.currentCluster = bestCluster
            if goldAmountAtCurrentPosition >= 50:
                self.agentState = AgentState.MINING

        elif self.agentState == AgentState.INCLUSTER:
            if goldAmountAtCurrentPosition >= 50:
                self.agentState = AgentState.MINING
            else:
                # print("bug1: ", self.currentCluster.total_gold)
                if self.currentCluster.total_gold <= 0:
                    # print("bug2")
                    self.agentState = AgentState.GOCLUSTER
                    self.currentCluster = None
        elif self.agentState == AgentState.MINING:
            if goldAmountAtCurrentPosition < 50:
                if self.currentCluster is not None and self.currentCluster.total_gold > 0:
                    self.agentState = AgentState.INCLUSTER
                else:
                    self.agentState = AgentState.GOCLUSTER

        # print("Current state:", self.agentState)
        # if(self.currentCluster != None):
        #     print("Total cluster gold: ", self.currentCluster.total_gold)

        ''' DO ACTION '''
        if self.agentState == AgentState.GOCLUSTER:
            desx, desy, bestCluster = self.findBestCluster()
            bestValue = 10000
            for action in actions:
                # print("\tTry action: ", action)
                posx, posy, energy = self.get_successor(action)
                pathCost = self.estimatePathCost(posx, posy, desx, desy)
                if pathCost < bestValue:
                    bestValue = pathCost
                    bestAction = action
                    energyOfBest = energy
                    goldPos = {"posx": desx, "posy": desy, "amount": 0}

        elif self.agentState == AgentState.INCLUSTER:
            bestValue = -1
            for action in actions:
                # print("\tTry action: ", action)
                posx, posy, energy = self.get_successor(action)
                value, gold = self.new_evaluationFunc(posx, posy)
                if value > bestValue:
                    bestValue = value
                    bestAction = action
                    energyOfBest = energy
                    goldPos = gold

        elif self.agentState == AgentState.MINING:
            bestAction = 5
            energyOfBest = self.state.energy - 5
            goldPos = {"posx": self.state.x, "posy": self.state.y,
                       "amount": self.estimateReceivedGold(self.state.x, self.state.y)}

        if not self.isSleeping and energyOfBest <= 0:
            self.isSleeping = True
            return 4, goldPos
        elif self.isSleeping:
            if bestAction != 5 and self.state.energy < 32:
                return 4, goldPos
            elif bestAction == 5 and self.state.energy < 28:
                return 4, goldPos
        self.isSleeping = False

        return bestAction, goldPos

    def estimatePathCost(self, startx, starty, endx, endy):
        hstep = 1 if endx > startx else -1
        vstep = 1 if endy > starty else -1
        i, j = startx, starty
        totalCostHL, totalCostLH = 0, 0
        # ngang roi doc
        hcost, vcost = 0, 0
        # print("ngang roi doc")
        while(True):
            # print("Debug", i, j, self.state.mapInfo.get_cell_cost(i, j))
            hcost += self.state.mapInfo.get_cell_cost(i, j)
            if i == endx:
                break
            i += hstep
        if j == endy:
            totalCostHL = hcost
        else:
            j += vstep
            while(True):
                # print("Debug", i, j, self.state.mapInfo.get_cell_cost(i, j))
                vcost += self.state.mapInfo.get_cell_cost(i, j)
                if j == endy:
                    break
                j += vstep
            totalCostHL = hcost + vcost

        # doc roi ngang
        # print("doc roi ngang")
        hcost, vcost = 0, 0
        i, j = startx, starty
        while(True):
            # print("Debug", i, j, self.state.mapInfo.get_cell_cost(i, j))
            vcost += self.state.mapInfo.get_cell_cost(i, j)
            if j == endy:
                break
            j += vstep
        if i == endx:
            totalCostLH = vcost
        else:
            i += hstep
            while(True):
                # print("Debug", i, j, self.state.mapInfo.get_cell_cost(i, j))
                hcost += self.state.mapInfo.get_cell_cost(i, j)
                if i == endx:
                    break
                i += hstep
            totalCostLH = hcost + vcost
        return min(totalCostHL, totalCostLH)

    def estimateBotPosition(self, goldX, goldY):
        countBot = []
        for player in self.state.players:
            if player["playerId"] != self.state.id:
                distanceToGold = self.mahattan(
                    goldX, goldY, player["posx"], player["posy"])
                if distanceToGold <= 3:
                    countBot.append(distanceToGold)
        return countBot

    def new_evaluationFunc(self, posx, posy):
        # if self.state.mapInfo.get_cell_cost(posx, posy) >= 40:
        #     return 50,

        maxGoldScore = 0
        goldPos = None
        ''' estimate gold '''
        for gold in self.currentCluster.goldArray:
            distance = self.mahattan(posx, posy, gold["posx"], gold["posy"])
            distance = min(10, distance)
            if distance < 3:
                goldScore = (10 - distance) * 150 + gold["amount"]
            else:
                countBot = self.estimateBotPosition(gold["posx"], gold["posy"])
                goldScore = (10 - distance) * 150 + \
                    gold["amount"] - 50 * \
                    (len(countBot) * distance - sum(countBot))
                goldScore = max(1, goldScore)
            if maxGoldScore < goldScore:
                goldPos = gold
                maxGoldScore = goldScore

        pathScore = self.estimatePathCost(
            posx, posy, goldPos["posx"], goldPos["posy"])
        # print("Goldscore:", maxGoldScore,
        #       goldPos["posx"], goldPos["posy"], goldPos["amount"])
        # print("PathScore:", pathScore)
        return maxGoldScore - pathScore * 30, goldPos

    def check_terminate(self):
        return self.state.status != State.STATUS_PLAYING
