from policies import base_policy as bp
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop
import random
from scipy.spatial.distance import cdist

# deep mind model
EPSILON = 0.05
from keras import layers
from keras.models import Model
#
import tensorflow as tf
import os
from keras.models import load_model

#os.environ['KERAS_BACKEND'] = 'tensorflow'


class Custom305248791(bp.Policy):

    def cast_string_args(self, policy_args):
#        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        return policy_args

    def init_run(self):
       
        #Hyper Parameters
        self.gamma = 0.8  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.001
        self.learning_rate = 0.001

        # State representation
        self.offset = 0
        self.window_size = 3
        self.window_length = 9
        self.num_board_features = 11
        self.num_old_features = 0
        self.num_of_special_features = 0
        self.state_size = self.num_old_features + self.num_of_special_features + self.num_board_features * self.window_size * self.window_length
        self.action_size = len(bp.Policy.ACTIONS)
        self.south_i = np.arange(-1 * self.offset, self.window_length - self.offset)
        self.south_j = np.array([-1, 0, 1])
        
        #Freezing Q related Parameters
        self.model = self.nn_deep_q_learning_model() 
        self.target_model = self.nn_deep_q_learning_model() 
        self.batch_size = 3 
        self.episodes = 1000
        self.freezing_duration = 1000
        
        self.replay_memory = deque(maxlen=300)
        self.error = deque(maxlen=300) # for Mem Prioritization

    """
    Helper Function for orienting the map features
    """
    def get_direction_vectors(self, direction):
        if direction == 'E': return [self.south_j, self.south_i]
        if direction == 'W': return [self.south_j, -1 * self.south_i]
        if direction == 'N': return [-1 * self.south_i, self.south_j]
        if direction == 'S': return [self.south_i, self.south_j]
    """
    Helper Function for orienting the map features
    """
    def get_feature_vectors(self, direction):
        if direction == 'E': return [self.south_j, self.south_i]
        if direction == 'W': return [self.south_j, self.south_i]
        if direction == 'N': return [self.south_i, self.south_j]
        if direction == 'S': return [self.south_i, self.south_j]

    """
    Creates the state representation
    flatten (window_length x window_width x map_features)
    """
    def preprocess_state(self, state):
        board, head = state
        rows, colls = self.board_size
        head_pos, direction = head
        r, c = head_pos
        dir_rows, dir_colls = self.get_direction_vectors(direction)
        feature_rows, feature_colls = self.get_feature_vectors(direction)
        features = np.zeros((len(dir_rows), len(dir_colls), self.num_board_features))
        ft_rows_min = min(feature_rows)
        ft_colls_min = min(feature_colls)
        for i_map, i_feature in zip(dir_rows, feature_rows):
            for j_map, j_feature in zip(dir_colls, feature_colls):
                features[i_feature - ft_rows_min, j_feature - ft_colls_min, board[
                    (r + i_map) % rows, (c + j_map) % colls] + 1] = self.window_length - np.asscalar(
                    cdist([[i_map, j_map]], [[0, 0]], metric='cityblock'))

        if (direction == 'E' or direction == 'W'):
            features = np.transpose(features, [1, 0, 2])
            if (direction == 'E'):
                features = np.flip(features, 1)
        elif (direction == 'N'):
            features = np.flip(features, 1)
        features = np.reshape(features, [1, self.state_size - self.num_of_special_features - self.num_old_features])
        return features

    """
    Creates sequential NN model
    """
    def nn_deep_q_learning_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(20, input_shape=(self.state_size,), activation='relu'))  # current  max score on 30
        model.add(Dense(20, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss="logcosh",
                      optimizer=RMSprop(lr=self.learning_rate))
        
        return model

    """
    Adds "memories" to the memory batch
    """
    def remember(self, state, action, reward, next_state, error):
        self.replay_memory.append((state, action, reward, next_state))
        self.error.append(error)
        
        
    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        if np.random.rand() < self.epsilon:
            return np.random.choice(bp.Policy.ACTIONS)
        
        new_state_features = self.preprocess_state(new_state)
        Q_value = self.model.predict(new_state_features)
        chosen_action = bp.Policy.ACTIONS[np.argmax(Q_value[0])]
        self.previous_action = chosen_action
        return chosen_action

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
    
        new_state_features = self.preprocess_state(new_state)
        prev_state_features = self.preprocess_state(prev_state)
        target = (reward + self.gamma *
                  np.amax(self.target_model.predict(new_state_features)[0]))
        target_f = np.amax(self.target_model.predict(prev_state_features)[0])
        error = abs(target - target_f) # calculating the the target error
        self.remember(prev_state_features, prev_action, reward, new_state_features, error)
        if len(self.replay_memory) > self.batch_size:
            self.deep_q_learn()
        if (round % self.freezing_duration):
            self.target_model.set_weights(self.model.get_weights())

    def deep_q_learn(self):
        #Chooses memories from the mem batch, the higher the error, 
        # the higher it's probability to get chosen, no duplicates
        minibatch = np.take(self.replay_memory,
                            np.random.choice(np.arange(len(self.replay_memory)), self.batch_size, replace=False,
                                             p=np.array(self.error) / np.sum(self.error)), axis=0)
        for state, action, reward, next_state in minibatch:
            target = (reward + self.gamma *
                      np.amax(self.target_model.predict(next_state)[0]))
            target_f = self.target_model.predict(state)
            target_f[0][bp.Policy.ACTIONS.index(action)] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

