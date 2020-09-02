#!/usr/bin/python3
from MINER_STATE import State
import numpy as np
import sys
from warnings import simplefilter
import numpy as np
import enum
import sys
import dbscanner
simplefilter(action='ignore', category=FutureWarning)


TreeID = 1
TrapID = 2
SwampID = 3


class AgentState(enum.Enum):
    MINING = 0
    GOCLUSTER = 1
    INCLUSTER = 2


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


class Best_2_4:
    def __init__(self, agentId=None, state=None):
        # self.socket = GameSocket(host, port)
        if agentId:
            self.agent_id = agentId
            self.info = PlayerInfo(self.agent_id)
        if state:
            self.state = state
        else:
            self.state = State()
        self.isSleeping = False
        self.swampCount = -1
        self.sleepCount = -1
        self.sleepBonus = [12, 16, 25]

        self.agentState = AgentState.GOCLUSTER
        # Storing the last score for designing the reward function
        self.score_pre = self.state.score
        self.currentCluster = None

        self.targetCluster = None
        self.targetDesx = -1
        self.targetDesy = -1
        self.countGoCluster = 0

    def start(self):  # connect to server
        self.socket.connect()

    def end(self):  # disconnect server
        self.socket.close()

    def send_map_info(self, request):  # tell server which map to run
        self.socket.send(request)

    def reset(self):  # start new game
        try:
            message = self.socket.receive()  # receive game info from server
            # print(message)
            self.state.init_state(message)  # init state
        except Exception as e:
            import traceback
            traceback.print_exc()

    def step(self, action):  # step process
        self.socket.send(action)  # send action to server
        try:
            message = self.socket.receive()  # receive new state from server
            # print("New state: ", message)
            self.state.update_state(message)  # update to local state
        except Exception as e:
            import traceback
            traceback.print_exc()

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

    # def estimateChange(self, i, j):
    #     cost = self.state.mapInfo.get_cell_cost(i, j)
    #     return cost

    def estimateReceivedGold(self, x, y):
        initGold = self.state.mapInfo.gold_amount(x, y)
        # print("Init gold:", initGold)
        if initGold <= 0:
            return 0
        countPlayer = 0
        for player in self.state.players:
            if player["playerId"] != self.state.id and player["posx"] == x and player["posy"] == y and player["energy"] > 5:
                countPlayer += 1

        if countPlayer == 0:
            return initGold
        if initGold >= (countPlayer + 1) * 50:
            return initGold
        return initGold // (countPlayer + 1)

    def distanceToCluster(self, cluster, posx, posy):
        distanceArray = list(map(lambda x: self.estimatePathCost(
            x['posx'], x['posy'], posx, posy), cluster.goldArray))
        minDistance = min(distanceArray)

        return minDistance, cluster.goldArray[distanceArray.index(minDistance)]['posx'], cluster.goldArray[distanceArray.index(minDistance)]['posy']

    def findBestCluster(self):
        bestCluster = None
        globalClusterScore = -1
        bestDestinationx, bestDestinationy = 0, 0
        for cluster in self.state.mapInfo.clusterList:
            if cluster.total_gold > 0:
                estimatePathToCluster, destinationx, destinationy = self.distanceToCluster(
                    cluster, self.state.x, self.state.y)
                mahattanDistance, _, _ = cluster.distanceToCluster(
                    self.state.x, self.state.y)
                estimateGoldInCluster = cluster.total_gold - 50 * \
                    cluster.checkEnermyInCluster(
                        self.state.players)*(mahattanDistance)
                clusterScore = estimateGoldInCluster - 10 * estimatePathToCluster
                clusterScore = max(0, clusterScore)
                if globalClusterScore < clusterScore:
                    globalClusterScore = clusterScore
                    bestDestinationx, bestDestinationy = destinationx, destinationy
                    bestCluster = cluster
        return bestDestinationx, bestDestinationy, bestCluster

    def findBestGoldMine(self):
        pass

    def get_action(self):
        if len(self.state.mapInfo.golds) == 0:
            return 4, {"posx": self.state.x, "posy": self.state.y, "amount": 0}
        bestValue = -10000
        bestAction = None
        energyOfBest = self.state.energy
        goldPos = None

        actions = self.legalAction()
        goldAmountAtCurrentPosition = self.estimateReceivedGold(
            self.state.x, self.state.y)

        ''' CHANGE STATE '''
        if self.agentState == AgentState.GOCLUSTER:
            # desx, desy, bestCluster = self.findBestCluster()
            if self.targetDesx == self.state.x and self.targetDesy == self.state.y:
                self.agentState = AgentState.INCLUSTER
                self.currentCluster = self.targetCluster
                self.targetCluster = None
                self.targetDesx, self.targetDesy = -1, -1
                self.countGoCluster = 0
            if goldAmountAtCurrentPosition >= 50:
                self.agentState = AgentState.MINING

        elif self.agentState == AgentState.INCLUSTER:
            if goldAmountAtCurrentPosition >= 50:
                self.agentState = AgentState.MINING
            else:
                print("bug1: ", self.currentCluster.total_gold)
                if self.currentCluster.total_gold <= 0:
                    print("bug2")
                    self.agentState = AgentState.GOCLUSTER
                    self.currentCluster = None
        elif self.agentState == AgentState.MINING:
            if goldAmountAtCurrentPosition < 50:
                # if self.currentCluster is not None and self.currentCluster.total_gold > 0:
                #     self.agentState = AgentState.INCLUSTER
                # else:
                #     self.agentState = AgentState.GOCLUSTER
                #     self.currentCluster = None
                newDesx, newDesy, newCluster = self.findBestCluster()
                if self.currentCluster is None or newCluster._id != self.currentCluster._id:
                    self.agentState = AgentState.GOCLUSTER
                    self.currentCluster = None
                else:
                    self.agentState = AgentState.INCLUSTER

        print("Current state:", self.agentState)
        if(self.currentCluster != None):
            print("Total cluster gold: ", self.currentCluster.total_gold)

        ''' DO ACTION '''
        if self.agentState == AgentState.GOCLUSTER:
            desx, desy, bestCluster = self.findBestCluster()
            if self.targetCluster != bestCluster:
                if self.countGoCluster <= 2 or self.targetCluster.total_gold < 50:
                    self.targetCluster = bestCluster
                    self.targetDesx, self.targetDesy = desx, desy
                    self.countGoCluster = 0
                else:
                    self.countGoCluster += 1
            else:
                self.countGoCluster += 1
            bestValue = 10000
            for action in actions:
                print("\tTry action: ", action)
                posx, posy, energy = self.get_successor(action)
                pathCost = self.estimatePathCost(
                    posx, posy, self.targetDesx, self.targetDesy)
                if pathCost < bestValue:
                    bestValue = pathCost
                    bestAction = action
                    energyOfBest = energy
                    goldPos = {"posx": desx, "posy": desy, "amount": 0}

        elif self.agentState == AgentState.INCLUSTER:
            bestValue = -10000
            for action in actions:
                print("\tTry action: ", action)
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

        # ''' check gold to dig '''
        # if self.estimateReceivedGold(self.state.x, self.state.y) >= 50:
        #     bestAction = 5
        #     energyOfBest = self.state.energy - 5
        #     goldPos = {"posx": self.state.x, "posy": self.state.y, "amount": self.estimateReceivedGold(self.state.x, self.state.y)}
        # else:
        #     for action in actions:
        #         print("\tTry action: ", action)
        #         posx, posy, energy = self.get_successor(action)
        #         value, gold = self.new_evaluationFunc(posx, posy)
        #         if value > bestValue:
        #             bestValue = value
        #             bestAction = action
        #             energyOfBest = energy
        #             goldPos = gold

        #         # print("-------------------------")
        # print("Best action:", bestAction, energyOfBest)

        # print("Best action:", bestAction)
        # need_energy = self.
        if not self.isSleeping and energyOfBest <= 0:
            self.isSleeping = True
            return 4, goldPos
        elif self.isSleeping:
            if self.state.energy < 36:
                return 4, goldPos
            # if bestAction != 5 and self.state.energy < 32:
            #     return 4, goldPos
            # elif bestAction == 5 and self.state.energy < 32:
            #     return 4, goldPos
        self.isSleeping = False

        return bestAction, goldPos

    # def new_estimatePathCost(self, startx, starty, endx, endy):
    #     hstep = 1 if endx > startx else -1
    #     vstep = 1 if endy > starty else -1
    #     j = endy

    #     costMatrix = self.state.mapInfo.map[min(starty, endy): max(starty, endy)][min(startx, endx): max(startx, endx)]
    #     dpCost = [[0]*abs(startx - endx) for i in range(abs(starty - endy))]

    #     i = endx - hstep
    #     while (i != startx) {

    #     }
    #     while(j != starty) {
    #         i = endx
    #         while(i != startx) {

    #         }
    #     }

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

        maxGoldScore = -10000
        goldPos = None
        ''' estimate gold '''
        for gold in self.currentCluster.goldArray:
            distance = self.mahattan(posx, posy, gold["posx"], gold["posy"])
            pathScoreToGold = self.estimatePathCost(
                posx, posy, gold["posx"], gold["posy"])
            distance = min(10, distance)
            if distance < 3:
                goldScore = gold["amount"] + self.neighborGold(
                    gold["posx"], gold["posy"]) - pathScoreToGold*25
            else:
                countBot = self.estimateBotPosition(gold["posx"], gold["posy"])
                goldScore = gold["amount"] + self.neighborGold(
                    gold["posx"], gold["posy"]) - pathScoreToGold*25 - 50*(len(countBot) * distance - sum(countBot))
                # goldScore = max(1, goldScore)
            if maxGoldScore < goldScore:
                goldPos = gold
                maxGoldScore = goldScore

        # pathScore = self.estimatePathCost(
        #     posx, posy, goldPos["posx"], goldPos["posy"])
        print("Goldscore:", maxGoldScore,
              goldPos["posx"], goldPos["posy"], goldPos["amount"])
        # print("PathScore:", pathScore)
        return maxGoldScore, goldPos

    def neighborGold(self, posx, posy):
        neighborGold = 0
        neighbors = [(posx+1, posy), (posx, posy+1),
                     (posx-1, posy), (posx, posy-1)]
        for neighbor in neighbors:
            neighborGold += self.state.mapInfo.gold_amount(
                neighbor[0], neighbor[1])
        return neighborGold*0.5

    def evaluationFunc(self, posX, posY, energy, golds, obstacles, params):
        def mahattan(x1, y1, x2, y2):
            return abs(x1-x2) + abs(y1 - y2)

        alpha = 0.95
        obstaclesDict = {}
        score = 0
        goldsArray = [(gold['posx'], gold['posy'])
                      for gold in golds if self.estimateReceivedGold(gold["posx"], gold["posy"]) > 10]
        for obstacle in obstacles:
            position = (obstacle['posx'], obstacle['posy'])
            obstaclesDict[position] = obstacle['type']

        # CHECK OBSTACLE
        penaltyScore = 0
        if (posX, posY) in obstaclesDict.keys():
            if obstaclesDict[(posX, posY)] == TreeID:
                # score -= 13
                penaltyScore = 13
            if obstaclesDict[(posX, posY)] == TrapID:
                # score -= 10
                penaltyScore = 10
            if obstaclesDict[(posX, posY)] == SwampID:
                # score -= self.swampPen[params['swampCount']]
                penaltyScore = self.swampPen[params['swampCount']]

        # if penaltyScore > 20:
        #     score -= 10

        goldPosX, goldPosY = -1, -1
        for gold in goldsArray:
            goldScore = alpha/(0.1+mahattan(posX, posY, gold[0], gold[1]))
            if goldScore > score:
                score = goldScore
            # if (5 * alpha/(0.1+mahattan(posX, posY, gold[0], gold[1])) + (1-alpha)*gold[2]) > score:
            #     score = (5 * alpha/(0.1+mahattan(posX, posY,
            #                                      gold[0], gold[1])) + (1-alpha)*gold[2])
                goldPosX = gold[0]
                goldPosY = gold[1]
        # print("Gold position:", goldPosX, goldPosY)
        # print("Self position:", self.state.x, self.state.y)
        # score = max(score, (5 * alpha/(0.1+mahattan(posX, posY,
        #                                             gold[0], gold[1])) + (1-alpha)*gold[2]))
        # print("Score, penaltyScore:", score, penaltyScore)
        return score
    # Functions are customized by client

    def get_state(self):
        # Building the map
        view = np.zeros([self.state.mapInfo.max_x + 1,
                         self.state.mapInfo.max_y + 1], dtype=int)
        for i in range(self.state.mapInfo.max_x + 1):
            for j in range(self.state.mapInfo.max_y + 1):
                if self.state.mapInfo.get_obstacle(i, j) == TreeID:  # Tree
                    view[i, j] = -TreeID
                if self.state.mapInfo.get_obstacle(i, j) == TrapID:  # Trap
                    view[i, j] = -TrapID
                if self.state.mapInfo.get_obstacle(i, j) == SwampID:  # Swamp
                    view[i, j] = -SwampID
                if self.state.mapInfo.gold_amount(i, j) > 0:
                    view[i, j] = self.state.mapInfo.gold_amount(i, j)

        DQNState = view.flatten().tolist()  # Flattening the map matrix to a vector

        # Add position and energy of agent to the DQNState
        DQNState.append(self.state.x)
        DQNState.append(self.state.y)
        DQNState.append(self.state.energy)

        # Add position of bots
        for player in self.state.players:
            if player["playerId"] != self.state.id:
                DQNState.append(player["posx"])
                DQNState.append(player["posy"])

        # Convert the DQNState from list to array
        DQNState = np.array(DQNState)

        return DQNState

    def check_terminate(self):
        return self.state.status != State.STATUS_PLAYING
