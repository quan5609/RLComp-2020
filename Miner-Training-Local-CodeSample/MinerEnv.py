import sys
import numpy as np
# in testing version, please use GameSocket instead of GAME_SOCKET_DUMMY
from GAME_SOCKET_DUMMY import GameSocket
from MINER_STATE import State


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
            self.state.init_state(message)  # init state
        except Exception as e:
            import traceback
            traceback.print_exc()

    def step(self, action):  # step process
        self.socket.send(action)  # send action to server
        try:
            message = self.socket.receive()  # receive new state from server
            self.state.update_state(message)  # update to local state
        except Exception as e:
            import traceback
            traceback.print_exc()

    ''' CUSTOMIZE STATE '''

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

        # Convert the DQNState from list to array for training
        DQNState = np.array(DQNState)

        return DQNState

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

    def get_successor(self, state, action):
        newX, newY, newEnergy = self.state.x, self.state.y, self.state.energy
        if action == 4:
            self.sleepCount += 1
            newEnergy += self.sleepBonus[self.sleepCount]
            newEnergy = min(50, newEnergy)
        else:
            self.sleepCount = -1
            if action == 0:
                newX -= 1
                newEnergy += self.estimateChange(newX, newY)[1]
            elif action == 1:
                newX += 1
                newEnergy += self.estimateChange(newX, newY)[1]
            elif action == 2:
                newY -= 1
                newEnergy += self.estimateChange(newX, newY)[1]
            elif action == 3:
                newY += 1
                newEnergy += self.estimateChange(newX, newY)[1]
            else:
                newEnergy -= 4
        return newX, newY, newEnergy, state

    def estimateChange(self, i, j):
        if self.state.mapInfo.get_obstacle(i, j) == TreeID:  # Tree
            return 0, -20
        if self.state.mapInfo.get_obstacle(i, j) == TrapID:  # Trap
            return 0, -10
        if self.state.mapInfo.get_obstacle(i, j) == SwampID:
            if self.swampCount < 3:
                self.swampCount += 1  # Swamp
            return 0, self.swampPen[self.swampCount]
        if self.state.mapInfo.gold_amount(i, j) > 0:
            return 0, -4
    ''' CUSTOMIZE REWARD '''

    def get_action(self):
        actions = self.legalAction(self.state)
        bestValue = 0
        bestAction = None

        if not self.isSleeping and self.state.energy < 5:
            self.isSleeping = True
            return 4
        elif self.isSleeping and self.state.energy < 50:
            return 4
        self.isSleeping = False
        if self.state.mapInfo.gold_amount(self.state.x, self.state.y) > 0:
            return 5

        for action in actions:
            posX, posY, energy, state = self.get_successor(state, action)
            value = self.evaluationFunc(
                posX, posY, energy, self.state.mapInfo.golds, self.state.mapInfo.obstacles)
            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction

    def evaluationFunc(self, posX, posY, energy, golds, obstacles):
        def mahattan(x1, y1, x2, y2):
            return abs(x1-x2) + abs(y1 - y2)

        alpha = 0.9

        goldsArray = [(gold['posx'], gold['posy'], gold['amount'])
                      for gold in golds]
        obstaclesArray = [(obstacle['posx'], obstacle['posy'])
                          for obstacle in obstacles]
        if (posX, posY) in obstaclesArray:
            return -9999

        score = 0
        for gold in goldsArray:
            score = max(score, alpha/(0.1+mahattan(posX, posY,
                                                   gold[0], gold[1])) + (1-alpha)*gold[3])

        return score

    def get_reward(self):
        # Calculate reward
        reward = 0
        score_action = self.state.score - self.score_pre
        self.score_pre = self.state.score
        if score_action > 0:
            # If the DQN agent crafts golds, then it should obtain a positive reward (equal score_action)
            reward += score_action

        reward += 0.01 * \
            self.state.mapInfo.gold_amount(self.state.x, self.state.y)
        reward += 0.2 * self.state.mapInfo.is_row_has_gold(self.state.y)
        reward += 0.2 * self.state.mapInfo.is_row_has_gold(self.state.x)

        # If the DQN agent crashs into obstacels (Tree, Trap, Swamp), then it should be punished by a negative reward
        # Tree
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == TreeID:
            reward -= TreeID
        # Trap
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == TrapID:
            reward -= TrapID
        # Swamp
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == SwampID:
            reward -= SwampID

        # If out of the map, then the DQN agent should be punished by a larger negative reward.
        if self.state.status == State.STATUS_ELIMINATED_WENT_OUT_MAP:
            reward += -10

        # Run out of energy, then the DQN agent should be punished by a larger negative reward.
        if self.state.status == State.STATUS_ELIMINATED_OUT_OF_ENERGY:
            reward += -10
        # print ("reward",reward)
        return reward

    def check_terminate(self):
        # Checking the status of the game
        # it indicates the game ends or is playing
        return self.state.status != State.STATUS_PLAYING
