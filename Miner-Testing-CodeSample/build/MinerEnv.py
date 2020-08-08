# in testing version, please use GameSocket instead of GameSocketDummy
#!/usr/bin/python3

from GAME_SOCKET import GameSocket
from MINER_STATE import State
import numpy as np
import sys
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


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
        self.swampPen = [-5, -20, -40, -100]
        # Storing the last score for designing the reward function
        self.score_pre = self.state.score

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

    def legalAction(self):
        temp_action = [0, 1, 2, 3]
        action = [0, 1, 2, 3]
        currX, currY = self.state.x, self.state.y
        newPos = [(currX - 1, currY), (currX + 1, currY),
                  (currX, currY - 1), (currX, currY + 1)]
        for index in range(len(newPos)):
            if self.state.mapInfo.get_cell_cost(newPos[index][0], newPos[index][1]) == 100:
                action.remove(temp_action[index])
        if self.state.x == self.state.mapInfo.max_x:
            action.remove(1)
        elif self.state.x == 0:
            action.remove(0)
        if self.state.y == self.state.mapInfo.max_y:
            action.remove(3)
        elif self.state.y == 0:
            action.remove(2)
        return action

    def get_successor(self, action, params):
        newX, newY, newEnergy = self.state.x, self.state.y, self.state.energy
        # if action == 4:
        #     self.sleepCount += 1
        #     newEnergy += self.sleepBonus[self.sleepCount]
        #     newEnergy = min(50, newEnergy)
        # else:
        #     self.sleepCount = -1
        if action == 0:
            newX -= 1
            newParams, deltaE = self.estimateChange(newX, newY, params)
            newEnergy += deltaE
        elif action == 1:
            newX += 1
            newParams, deltaE = self.estimateChange(newX, newY, params)
            newEnergy += deltaE
        elif action == 2:
            newY -= 1
            newParams, deltaE = self.estimateChange(newX, newY, params)
            newEnergy += deltaE
        elif action == 3:
            newY += 1
            newParams, deltaE = self.estimateChange(newX, newY, params)
            newEnergy += deltaE
        else:
            newEnergy -= 4
        return newX, newY, newEnergy, newParams

    def estimateChange(self, i, j, params):
        if self.state.mapInfo.gold_amount(i, j) > 0:
            return params, -4
        typeOb, penaltyOb = self.state.mapInfo.get_obstacle_and_penalty(i, j)

        # for RLCOMP
        # if typeOb == TreeID:
        #     return params, -20

        # for Test:
        # if typeOb == TreeID:
        #     return params, -3
        # if typeOb == TrapID:
        #     return params, -2
        # if typeOb == SwampID:
        #     return params, -3
        # return params, penaltyOb

        if self.state.mapInfo.get_obstacle(i, j) == TreeID:  # Tree
            return params, -20
        if self.state.mapInfo.get_obstacle(i, j) == TrapID:  # Trap
            return params, -10
        if self.state.mapInfo.get_obstacle(i, j) == SwampID:
            if self.swampCount < 3:
                self.swampCount += 1  # Swamp
            return params, self.swampPen[self.swampCount]
        if self.state.mapInfo.gold_amount(i, j) > 0:
            return params, -4
        return params, -1

    def estimateReceivedGold(self, x, y):
        # print("Gold Array:", self.state.mapInfo.golds)
        # print("pos:", x, y)
        initGold = self.state.mapInfo.gold_amount(x, y)
        # print("Init gold:", initGold)
        if initGold <= 0:
            return 0
        countPlayer = 0
        for player in self.state.players:
            if player["playerId"] != self.state.id:
                if player["posx"] == x and player["posy"] == y and player["energy"] > 5:
                    countPlayer += 1

        if countPlayer == 0:
            return initGold
        if initGold >= countPlayer * 50:
            return initGold
        return initGold // countPlayer

    def get_action(self):
        actions = self.legalAction()
        params = {"swampCount": self.swampCount}
        bestValue = -10000
        bestAction = None
        energyOfBest = self.state.energy

        # if self.state.mapInfo.gold_amount(self.state.x, self.state.y) > 0:
        # print("Estimate Gold:", self.estimateReceivedGold(
        #     self.state.x, self.state.y))

        ''' check gold to dig '''
        if self.estimateReceivedGold(self.state.x, self.state.y) >= 50:
            bestAction = 5
            energyOfBest = self.state.energy - 5
        else:
            for action in actions:
                print("try action: ", action)
                posx, posy, energy, params = self.get_successor(action, params)
                value = self.new_evaluationFunc(posx, posy)
                # value = self.evaluationFunc(
                #     posX, posY, energy, self.state.mapInfo.golds, self.state.mapInfo.obstacles, params)
                if value > bestValue:
                    bestValue = value
                    bestAction = action
                    energyOfBest = energy
                # print("-------------------------")
        print("Best action:", bestAction, energyOfBest)

        # print("Best action:", bestAction)

        if not self.isSleeping and energyOfBest <= 0:
            self.isSleeping = True
            return 4
        elif self.isSleeping and self.state.energy < 40:
            return 4
        self.isSleeping = False

        return bestAction

    def estimatePathCost(self, startx, starty, endx, endy):
        print(self.state.mapInfo.map)
        hstep = 1 if endx > startx else -1
        vstep = 1 if endy > starty else -1
        i, j = startx, starty
        totalCostHL, totalCostLH = 0, 0
        # ngang roi doc
        hcost, vcost = 0, 0
        print("ngang roi doc")
        while(True):
            print("Debug", i, j, self.state.mapInfo.get_cell_cost(i, j))
            hcost += self.state.mapInfo.get_cell_cost(i, j)
            if i == endx:
                break
            i += hstep
        if j == endy:
            totalCostHL = hcost
        else:
            j += vstep
            while(True):
                print("Debug", i, j, self.state.mapInfo.get_cell_cost(i, j))
                vcost += self.state.mapInfo.get_cell_cost(i, j)
                if j == endy:
                    break
                j += vstep
            totalCostHL = hcost + vcost

        # doc roi ngang
        print("doc roi ngang")
        hcost, vcost = 0, 0
        i, j = startx, starty
        while(True):
            print("Debug", i, j, self.state.mapInfo.get_cell_cost(i, j))
            vcost += self.state.mapInfo.get_cell_cost(i, j)
            if j == endy:
                break
            j += vstep
        if i == endx:
            totalCostLH = vcost
        else:
            i += hstep
            while(True):
                print("Debug", i, j, self.state.mapInfo.get_cell_cost(i, j))
                hcost += self.state.mapInfo.get_cell_cost(i, j)
                if i == endx:
                    break
                i += hstep
            totalCostLH = hcost + vcost

        return min(totalCostHL, totalCostLH)

    def new_evaluationFunc(self, posx, posy):
        def mahattan(x1, y1, x2, y2):
            return abs(x1-x2) + abs(y1 - y2)

        if self.state.mapInfo.get_cell_cost(posx, posy) >= 40:
            return 50
        elif self.state.mapInfo.get_cell_cost(posx, posy) >= 50:
            return 5  # no hope

        maxGoldScore = 0
        goldPos = None
        ''' estimate gold '''
        for gold in self.state.mapInfo.golds:
            distance = mahattan(posx, posy, gold["posx"], gold["posy"])
            distance = 10 if distance > 10 else distance
            goldScore = (10 - distance) * 150 + gold["amount"]
            if maxGoldScore < goldScore:
                goldPos = gold
                maxGoldScore = goldScore

        pathScore = self.estimatePathCost(
            posx, posy, goldPos["posx"], goldPos["posy"])
        print("Goldscore:", maxGoldScore,
              goldPos["posx"], goldPos["posy"], goldPos["amount"])
        print("PathScore:", pathScore)
        return maxGoldScore - pathScore * 30

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
