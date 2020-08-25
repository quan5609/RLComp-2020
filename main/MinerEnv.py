# in testing version, please use GameSocket instead of GameSocketDummy
#!/usr/bin/python3

from GAME_SOCKET_DUMMY import GameSocket
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


class MinerEnv:
    def __init__(self, host, port):
        self.socket = GameSocket(host, port)
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
        self.clusterNum = 0
        self.sorted_cluster_list = None

    def start(self):  # connect to server
        self.socket.connect()

    def end(self):  # disconnect server
        self.socket.close()

    def send_map_info(self, request):  # tell server which map to run
        self.socket.send(request)

    def reset(self):  # start new game
        try:
            # self.socket.reset()
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

    def estimateReceivedGold(self, x, y):
        initGold = self.state.mapInfo.gold_amount(x, y)
        # print("Init gold:", initGold)
        if initGold <= 0:
            return 0
        countPlayer = 0
        for player in self.state.players:
            # if player["playerId"] != self.state.id and player["posx"] == x and player["posy"] == y and player["energy"] > 5:
            if player["playerId"] != self.state.id and player["posx"] == x and player["posy"] == y:
                countPlayer += 1

        if countPlayer == 0:
            return initGold
        if initGold >= (countPlayer + 1) * 50:
            return initGold
        return initGold // (countPlayer + 1)

    def distanceToCluster(self, cluster, posx, posy):
        distanceArray = list(map(lambda x: self.new_estimatePathCost(
            x['posx'], x['posy'], posx, posy), cluster.goldArray))
        minDistance = min(distanceArray)

        return minDistance, cluster.goldArray[distanceArray.index(minDistance)]['posx'], cluster.goldArray[distanceArray.index(minDistance)]['posy']

    def getClusterByIndex(self, clusterList, id):
        # print(self.clusterNum, id)
        _, destinationx, destinationy = self.distanceToCluster(
            self.sorted_cluster_list[id], self.state.x, self.state.y)
        return destinationx, destinationy, self.sorted_cluster_list[id]

    def findBestCluster(self, agentCluster):
        if agentCluster is not None and agentCluster < self.clusterNum:
            # print("BUG", agentCluster, self.clusterNum)
            return self.getClusterByIndex(self.state.mapInfo.clusterList, agentCluster)
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

    def getClusterState(self, cluster):
        if cluster.total_gold <= 0:
            return [0,0,0,0,0]

        x, y = cluster.center_x, cluster.center_y
        terrainCost = 0
        for i in range(y - 2, y + 3):
            for j in range(x - 2, x + 3):
                checkCell = self.state.mapInfo.get_cell_cost(i, j)
                checkObj = self.state.mapInfo.get_obstacle(i, j)
                if checkCell != -1 and checkObj != -1:
                    if checkObj == 1: # tree
                        terrainCost += 1
                    elif checkObj == 3: # swarm
                        terrainCost += 2
        pathCost = self.new_estimatePathCost(
            self.state.x, self.state.y, cluster.center_x, cluster.center_y)
        # if pathCost + 1 == 0:
        #     print("BUG:", )
        # try:
        result = [np.log(cluster.total_gold+1), cluster.center_x, cluster.center_y, np.log(pathCost+1), terrainCost]
        # except:
        # print("Bug log: pathcost", pathCost)
        # print("Bug log: cluster total gold", cluster.total_gold)
        # return [np.log(cluster.total_gold+1), cluster.center_x, cluster.center_y, np.log(pathCost+1)]
        return result

    def get_state(self):
        # Player State
        player_state = [self.state.x, self.state.y, self.state.energy, self.state.lastAction]
        # print("BUG", self.state.players)
        for player in self.state.players:
            # print("Debug state bot:", player)
            player_state += [player['posx'], player['posy'], player['energy'], player['lastAction']]

        if len(player_state) < 16:
            player_state += [0] * (16 - len(player_state))
        # Cluster state
        cluster_state = []
        self.sorted_cluster_list = sorted(
            self.state.mapInfo.clusterList, key=lambda x: x.total_gold, reverse=True)
        self.clusterNum = len(self.sorted_cluster_list)
        for i in range(8):
            if i < self.clusterNum:
                cluster_state += self.getClusterState(
                    self.sorted_cluster_list[i])
            else:
                cluster_state += [0, 0, 0, 0, 0]

        if self.targetCluster is None:
            target_cluster_state = [0, 0, 0, 0, 0]
        else:
            target_cluster_state = self.getClusterState(self.targetCluster)

        if self.currentCluster is None:
            current_cluster_state = [0, 0, 0, 0, 0]
        else:
            current_cluster_state = self.getClusterState(self.currentCluster)

        # print("Player State:", player_state)
        # print("Curent State:", current_cluster_state)
        # print("Target State:", target_cluster_state)
        # print("Cluster State:", cluster_state)

        DQNState = player_state + cluster_state + \
            target_cluster_state + current_cluster_state
        return np.array(DQNState)

    def get_reward(self):
        delta = self.state.score - self.score_pre
        self.score_pre = self.state.score
        if delta == 0:
            delta -= 5
        return 5 * delta

    def check_mining(self):
        goldAmountAtCurrentPosition = self.estimateReceivedGold(
            self.state.x, self.state.y)
        if goldAmountAtCurrentPosition >= 50:
            self.agentState = AgentState.MINING
            return self.agentState

    def get_agent_state(self,  agentCluster=None):
        if len(self.state.mapInfo.golds) == 0:
            return 4, {"posx": self.state.x, "posy": self.state.y, "amount": 0}
        goldAmountAtCurrentPosition = self.estimateReceivedGold(
            self.state.x, self.state.y)
        if goldAmountAtCurrentPosition >= 50:
            self.agentState = AgentState.MINING
            return self.agentState

        ''' CHANGE STATE '''
        if self.agentState == AgentState.GOCLUSTER:
            if self.targetCluster is None:
                newDesx, newDesy, newCluster = self.findBestCluster(
                    agentCluster)
                self.targetCluster = newCluster
                self.targetDesx, self.targetDesy = newDesx, newDesy

            if self.targetDesx == self.state.x and self.targetDesy == self.state.y:
                self.agentState = AgentState.INCLUSTER
                self.currentCluster = self.targetCluster
                self.targetCluster = None
                self.targetDesx, self.targetDesy = -1, -1
                self.countGoCluster = 0

            if goldAmountAtCurrentPosition >= 50:
                self.agentState = AgentState.MINING

        elif self.agentState == AgentState.INCLUSTER:
            newDesx, newDesy, newCluster = self.findBestCluster(agentCluster)
            if newCluster._id != self.currentCluster._id:
                self.agentState = AgentState.GOCLUSTER
                self.currentCluster = None
                self.targetCluster = newCluster
                self.targetDesx, self.targetDesy = newDesx, newDesy

            if goldAmountAtCurrentPosition >= 50:
                self.agentState = AgentState.MINING
            # else:
            #     if self.currentCluster.total_gold <= 0:
            #         self.agentState = AgentState.GOCLUSTER
            #         self.currentCluster = None
        elif self.agentState == AgentState.MINING:
            if goldAmountAtCurrentPosition < 50:
                newDesx, newDesy, newCluster = self.findBestCluster(
                    agentCluster)
                if self.currentCluster is None or newCluster._id != self.currentCluster._id:
                    self.agentState = AgentState.GOCLUSTER
                    self.currentCluster = None
                    self.targetCluster = newCluster
                    self.targetDesx, self.targetDesy = newDesx, newDesy
                else:
                    self.agentState = AgentState.INCLUSTER

        # print("Current state:", self.agentState)
        # if(self.currentCluster != None):
        #     print("Total cluster gold: ", self.currentCluster.total_gold)
        return self.agentState

    def get_action(self):
        bestValue = -10000
        bestAction = None
        energyOfBest = self.state.energy
        goldPos = None

        actions = self.legalAction()
        ''' DO ACTION '''
        if self.agentState == AgentState.GOCLUSTER:
            # desx, desy, bestCluster = self.findBestCluster(agentCluster)
            # if self.targetCluster != bestCluster:
            #     if self.countGoCluster <= 2 or self.targetCluster.total_gold < 50:
            #         self.targetCluster = bestCluster
            #         self.targetDesx, self.targetDesy = desx, desy
            #         self.countGoCluster = 0
            #     else:
            #         self.countGoCluster += 1
            # else:
            #     self.countGoCluster += 1
            bestValue = 10000
            for action in actions:
                # print("\tTry action: ", action)
                posx, posy, energy = self.get_successor(action)
                pathCost = self.new_estimatePathCost(
                    posx, posy, self.targetDesx, self.targetDesy)
                if pathCost < bestValue:
                    bestValue = pathCost
                    bestAction = action
                    energyOfBest = energy
                    # goldPos = {"posx": self.targetDesx,
                    #            "posy": self.targetDesx, "amount": 0}

        elif self.agentState == AgentState.INCLUSTER:
            bestValue = -1000000
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
            # goldPos = {"posx": self.state.x, "posy": self.state.y,
            #            "amount": self.estimateReceivedGold(self.state.x, self.state.y)}

        
        if self.isSleeping:
            if self.state.energy >= 36 and energyOfBest > 0:
                self.isSleeping = False
                return bestAction, goldPos
            return 4, goldPos
        else:
            if energyOfBest <= 0:
                self.isSleeping = True
                return 4, goldPos
            return bestAction, goldPos

    def new_estimatePathCost(self, startx, starty, endx, endy):
        hstep = 1 if endx > startx else -1
        vstep = 1 if endy > starty else -1

        if startx == endx and starty == endy:
            return 0

        if startx == endx:
            cost = 0
            idx = endy
            while(idx != starty):
                cost += self.state.mapInfo.map[idx][startx]
                idx = idx - vstep
            return cost

        if starty == endy:
            cost = 0
            idx = endx
            while(idx != startx):
                cost += self.state.mapInfo.map[starty][idx]
                idx = idx - hstep
            return cost

        # costMatrix = self.state.mapInfo.map[min(starty, endy): max(starty, endy)][min(startx, endx): max(startx, endx)]
        # dpCost = [[0]*abs(startx - endx) for i in range(abs(starty - endy))]
        costMatrix = self.state.mapInfo.map
        dpCost = [[0]*(self.state.mapInfo.max_x+1) for i in range(self.state.mapInfo.max_y+1)]

        i = endx - hstep
        j = endy
        while ((startx < endx and i >= startx) or (startx >= endx and i <= startx)):
            dpCost[j][i] = dpCost[j][i+hstep] + costMatrix[j][i+hstep]
            i = i - hstep
        i = endx
        j = endy - vstep
        while ((starty < endy and j >= starty) or (starty >= endy and j <= starty)):
            dpCost[j][i] = dpCost[j+vstep][i] + costMatrix[j+vstep][i]
            j = j - vstep
        
        j = endy - vstep
        while ((starty < endy and j >= starty) or (starty >= endy and j <= starty)):
            i = endx - hstep
            while ((startx < endx and i >= startx) or (startx >= endx and i <= startx)):
                dpCost[j][i] = min(dpCost[j][i+hstep] + costMatrix[j][i+hstep], dpCost[j+vstep][i] + costMatrix[j+vstep][i])
                i = i - hstep
            j = j - vstep
        return dpCost[starty][startx]

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
            pathScoreToGold = self.new_estimatePathCost(
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
        # print("Goldscore:", maxGoldScore,
        #       goldPos["posx"], goldPos["posy"], goldPos["amount"])
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

    def check_terminate(self):
        return self.state.status != State.STATUS_PLAYING
