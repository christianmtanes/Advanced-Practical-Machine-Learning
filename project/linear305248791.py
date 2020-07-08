from policies import base_policy as bp
import numpy as np
from scipy.spatial.distance import cdist
EPSILON = 0.05

class Linear305248791(bp.Policy):

    def cast_string_args(self, policy_args):
#        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        return policy_args
    def init_run(self):
        self.r_sum = 0
        self.num_features = 11
        self.theta = 0.1 * np.ones(5 * 5 * self.num_features + 1)
        self.alpha = 0.01
        self.gamma = 0.5
        self.epsilon = 1
        self.epsilon_decay = 0.01
    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        if np.random.rand() < self.epsilon:
            return np.random.choice(bp.Policy.ACTIONS)
        else:
            action = self.max_Qs(new_state)[1]
        return action
    
    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):     
        Q_sa = self.Q(prev_state, prev_action)
        max_Q = self.max_Qs(new_state)[0]
        delta = reward + self.gamma * max_Q - Q_sa
        gradient  = self.gradient_Q(prev_state, prev_action)
        self.theta = self.theta + self.alpha * delta * gradient
        self.epsilon -= self.epsilon_decay
        
    def Q(self, state, action):
        return np.dot(self.theta.T, self.features(state, action))
    
    def gradient_Q(self, state, action):
        return self.features(state, action)
    
    def max_Qs(self, new_state):
        max_q = -np.inf
        max_action = None
        for action in bp.Policy.ACTIONS:
            q = self.Q(new_state, action)
            if q > max_q:
                max_q = q
                max_action = action
        return max_q, max_action
    
    def features(self, state, action):
        board, head = state
        rows, colls = board.shape
        head_pos, direction = head
        next_position = head_pos.move(bp.Policy.TURNS[direction][action])
        r = next_position[0]
        c = next_position[1]
        features = np.zeros((5,5,self.num_features))
        for i in [-2,-1,0,1,2]:
            for j in [-2,-1,0,1,2]:
                features[i + 1, j + 1, board[(r + i) % rows, (c + j) % colls] + 1] = 4 - np.asscalar(cdist([[i,j]],[[0,0]],metric='cityblock'))      
        return np.append([1], features.flatten())