# -*- coding: utf-8 -*-
from random import random, randrange
import tensorflow as tf
import keras
from keras import backend as K
from keras import optimizers
from keras.layers import Dense, Activation
from keras.models import model_from_json
from keras.models import Sequential
import numpy as np
import math
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


# Deep Q Network off-policy
class DQN:

    def __init__(
            self,
            input_dim,  # The number of inputs for the DQN network
            action_space,  # The number of actions for the DQN network
            gamma=0.99,  # The discount factor
            epsilon=1,  # Epsilon - the exploration factor
            epsilon_min=0.01,  # The minimum epsilon
            epsilon_decay=0.99,  # The decay epislon for each update_epsilon time
            learning_rate=0.0002,  # The learning rate for the DQN network
            tau=0.125,  # The factor for updating the DQN target network from the DQN network
            model=None,  # The DQN model
            target_model=None,  # The DQN target model
            sess=None

    ):
        # Tensorflow GPU optimization
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)
        K.set_session(sess)
        self.sess.run(tf.compat.v1.global_variables_initializer())

        self.input_dim = input_dim
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.tau = tau

        # Creating networks
        # self.model = self.create_model()  # Creating the DQN model
        self.model = self.load_model()
        self.target_model = self.load_model(
            target=True)  # Creating the DQN target model

    def load_model(self, target=False):
        # load json and create model
        if target:
            name = 'DQNmodel_target_latest'
        else:
            name = 'DQNmodel_latest'
        json_file = open(
            'TrainedModels/' + name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # print(loaded_model.get_weights())
        # load weights into new model
        loaded_model.load_weights(
            'TrainedModels/' + name + '.h5')
        # print(loaded_model.get_weights())
        # model = self.create_model()
        # print(loaded_model.summary())
        # model = model.load_weights('/home/khmt/nddung/RLComp-2020/1234/main/TrainedModels/DQNmodel_20200816-0902_ep2400.h5')
        sgd = optimizers.SGD(lr=self.learning_rate, decay=1e-6, momentum=0.95)
        loaded_model.compile(optimizer=sgd,
                             loss='mse')
        print(loaded_model.summary())
        return loaded_model

    def create_model(self):
        def get_residual_block(inp, filter=64, is_max_pool=True):
            conv1 = keras.layers.Conv1D(
                filter, kernel_size=5, activation=keras.activations.relu, padding="same")(inp)
            conv2 = keras.layers.Conv1D(
                filter, kernel_size=5, activation=keras.activations.relu, padding="same")(conv1)
            skip_connect = keras.layers.Add()([inp, conv2])
            relu2 = keras.layers.ReLU()(skip_connect)
            if is_max_pool:
                max_pool = keras.layers.MaxPool1D(pool_size=2)(relu2)
                batch_norm = keras.layers.BatchNormalization(
                    trainable=True, epsilon=1e-4)(max_pool)
            else:
                batch_norm = keras.layers.BatchNormalization(
                    trainable=True, epsilon=1e-4)(relu2)

            return batch_norm
        # Creating the network
        # Two hidden layers (300,300), their activation is ReLu
        # One output layer with action_space of nodes, activation is linear.

        inp = keras.layers.Input(shape=(106, 1))
        conv1 = keras.layers.Conv1D(
            64, kernel_size=5, activation=keras.activations.relu, padding="same")(inp)
        res_block1 = get_residual_block(conv1)
        res_block2 = get_residual_block(res_block1, 64, False)
        res_block3 = get_residual_block(res_block2, filter=64)
        res_block4 = get_residual_block(res_block3, 64, False)
        res_block5 = get_residual_block(res_block4, filter=64)

        glopal_pool = keras.layers.GlobalMaxPool1D()(res_block5)
        dense1 = keras.layers.Dense(
            64, activation=keras.activations.relu)(glopal_pool)

        dense2 = keras.layers.Dense(
            128, activation=keras.activations.relu)(dense1)
        # dropout2 = keras.layers.Dropout(rate=0.2)(dense2)
        dense3 = keras.layers.Dense(16, activation='linear')(dense2)
        model = keras.models.Model(inputs=inp, outputs=dense3)
        model.summary()
        # adam = optimizers.adam(lr=self.learning_rate)
        sgd = optimizers.SGD(lr=self.learning_rate, decay=1e-6, momentum=0.95)
        adam = optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=adam, loss='mse')
        return model

    def act(self, state):
        # Get the index of the maximum Q values
        predicted = self.model.predict(state.reshape(1, len(state), 1))
        a_max = np.argmax(predicted)
        # print("Predict: ", predicted)
        if (random() < self.epsilon):
            a_chosen = randrange(self.action_space)
        else:
            a_chosen = a_max
        return a_chosen

    def DQN_predict(self, state):
        # Get the index of the maximum Q values
        return self.model.predict(state.reshape(1, len(state), 1))

    def replay(self, samples, batch_size):
        # print("######################################")
        inputs = np.zeros((batch_size, self.input_dim))
        targets = np.zeros((batch_size, self.action_space))
        currents = np.zeros((batch_size, self.action_space))

        print("Training....")
        for i in range(0, batch_size):
            state = samples[0][i, :]
            action = samples[1][i]
            reward = samples[2][i]
            new_state = samples[3][i, :]
            done = samples[4][i]

            inputs[i, :] = state
            targets[i, :] = self.target_model.predict(
                state.reshape(1, len(state), 1))
            currents[i, :] = self.model.predict(
                state.reshape(1, len(state), 1))
            # print("state:", state, action)
            # print("curr:", self.model.predict(
            #     state.reshape(1, len(state), 1)))
            if done:
                # if terminated, only equals reward
                targets[i, action] = reward
            else:
                Q_future = np.max(self.target_model.predict(
                    new_state.reshape(1, len(new_state), 1)))
                targets[i, action] = reward + Q_future * self.gamma
            if i in [0, 64, 128]:
                print("Target:", targets[i], reward)
        # Training
        # print("INPUT", inputs, inputs.shape)
        # print("TARGET", targets, targets.shape)
        # print("Current Q", currents)

        loss = self.model.train_on_batch(np.array(inputs).reshape(
            batch_size, self.input_dim, 1), np.array(targets).reshape(batch_size, self.action_space))
        print("Loss:", loss)
        if math.isnan(loss):
            return False
        return True

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(0, len(target_weights)):
            target_weights[i] = weights[i] * self.tau + \
                target_weights[i] * (1 - self.tau)

        self.target_model.set_weights(target_weights)

    def update_epsilon(self):
        self.epsilon = self.epsilon*self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

    def save_model(self, path, model_name):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(path + model_name + ".json", "w") as json_file:
            json_file.write(model_json)
            # serialize weights to HDF5
            self.model.save_weights(path + model_name + ".h5")
            print("Saved model to disk")

    def save_target_model(self, path, model_name):
        # serialize model to JSON
        model_json = self.target_model.to_json()
        with open(path + model_name + ".json", "w") as json_file:
            json_file.write(model_json)
            # serialize weights to HDF5
            self.target_model.save_weights(path + model_name + ".h5")
            print("Saved model to disk")
