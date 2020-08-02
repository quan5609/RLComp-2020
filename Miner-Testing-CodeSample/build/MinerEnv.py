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
            print(message)
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
            print("Current gold and energy:",
                  self.state.score, self.state.energy)

        except Exception as e:
            import traceback
            traceback.print_exc()

    def legalAction(self, state):
        action = [0, 1, 2, 3]
        if self.state.x == self.state.mapInfo.max_x:
            action.remove(1)
        elif self.state.x == 0:
            action.remove(0)
        if self.state.y == self.state.mapInfo.max_y:
            action.remove(3)
        elif self.state.y == 0:
            action.remove(2)
        # if self.state.energy <= 4 or self.state.mapInfo.gold_amount(self.state.x, self.state.y) == 0:
        #     action.remove(5)
        return action

    def get_successor(self, state, action, params):
        newX, newY, newEnergy = self.state.x, self.state.y, self.state.energy
        if action == 4:
            self.sleepCount += 1
            newEnergy += self.sleepBonus[self.sleepCount]
            newEnergy = min(50, newEnergy)
        else:
            self.sleepCount = -1
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
        if self.state.mapInfo.get_obstacle(i, j) == TreeID:  # Tree
            return params, -20
        if self.state.mapInfo.get_obstacle(i, j) == TrapID:  # Trap
            return params, -10
        if self.state.mapInfo.get_obstacle(i, j) == SwampID:
            if params['swampCount'] < 3:
                params['swampCount'] += 1  # Swamp
            return params, self.swampPen[self.swampCount]
        if self.state.mapInfo.gold_amount(i, j) > 0:
            return params, -4
        return params, -1

    def estimateReceivedGold(self, x, y):
        print("Gold Array:", self.state.mapInfo.golds)
        print("pos:", x, y)
        initGold = self.state.mapInfo.gold_amount(x, y)
        print("Init gold:", initGold)
        if initGold <= 0:
            return 0
        countPlayer = 0
        for player in self.state.players:
            if player["playerId"] != self.state.id:
                if player["posx"] == x and player["posy"] == y:
                    countPlayer += 1

        if countPlayer == 0:
            return initGold
        if initGold >= countPlayer * 50:
            return initGold
        return initGold // countPlayer

    def get_action(self):
        actions = self.legalAction(self.state)
        params = {"swampCount": self.swampCount}
        bestValue = -10000
        bestAction = None
        energyOfBest = -100

        # if self.state.mapInfo.gold_amount(self.state.x, self.state.y) > 0:
        # print("Estimate Gold:", self.estimateReceivedGold(
        #     self.state.x, self.state.y))
        if self.estimateReceivedGold(self.state.x, self.state.y) >= 50:
            bestAction = 5
            energyOfBest = self.state.energy - 5
        else:
            for action in actions:
                posX, posY, energy, params = self.get_successor(
                    self.state, action, params)
                value = self.evaluationFunc(
                    posX, posY, energy, self.state.mapInfo.golds, self.state.mapInfo.obstacles, params)
                if value > bestValue:
                    bestValue = value
                    bestAction = action
                    energyOfBest = energy
                print("-------------------------")
            print("Best action:", bestAction, energyOfBest)
            print(
                "###########################################################################")
        # print("Best action:", bestAction)

        if not self.isSleeping and energyOfBest <= 0:
            self.isSleeping = True
            return 4
        elif self.isSleeping and self.state.energy < 48:
            return 4
        self.isSleeping = False

        return bestAction

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
        print("Gold position:", goldPosX, goldPosY)
        print("Self position:", self.state.x, self.state.y)
        # score = max(score, (5 * alpha/(0.1+mahattan(posX, posY,
        #                                             gold[0], gold[1])) + (1-alpha)*gold[2]))
        print("Score, penaltyScore:", score, penaltyScore)
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
