# Use target network
import tensorflow as tf
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras import Input
from keras.models import Sequential, load_model
import random
import numpy as np 
from collections import deque

import caro_part5 as caro
from evaluate import evaluate
EPOSIDES = 1000
ALPHA = 0.001
GAMMA = 0.95
EPSILON = 0.7590483508202912
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.005

BOARD_SIZE = 20
BATCH_SIZE = 64

def rewards(board, player):
    if caro.check_win(board, player):
        return 1
    elif caro.check_win(board, -player):
        return -1
    else:
        return 0

# tim phan tu co vi tri hop le va co xac suat du doan cao nhat
def probability_positions(board, prediction):
    # emp = caro.empty_cells_around(board,2)
    # if len(emp) <=0:
    emp = caro.empty_cells(board)
    probs = []
    for row, col in emp:
        position = row * BOARD_SIZE + col
        prob_position = prediction[position] 
        probs.append([row, col, prob_position])
    sort_probs = sorted(probs, key = lambda x: x[2], reverse=True)
    max_prob = sort_probs[0]
    return max_prob[0], max_prob[1]

class DQNAgent:
    def __init__(self, input_shape,alpha, gamma, epsilon, epsilon_min, epsilon_decay, num_actions, memory):
        self.input_shape = input_shape
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.num_actions = num_actions
        self.memory = memory
        self.gobal_step = 0
        self.num_update_target_model = 0
        self.model = self.built_model()
        self.target_model = self.built_model()

    def built_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3,3), activation = 'relu', input_shape = self.input_shape))
        model.add(MaxPooling2D((2,2)))
        model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu'))
        model.add(MaxPooling2D((2,2)))
        model.add(Conv2D(128, kernel_size=(3,3), activation = 'relu'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.num_actions, activation = 'softmax'))
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model


    def action(self, state):
        r = random.random()
        if r <= self.epsilon:
            # able_cells = caro.empty_cells_around(state, 2)
            # if len(able_cells) > 0:
            #     row, col = random.sample(able_cells, 1)[0]
            # else:
            able_cells = caro.empty_cells(state)
            row, col = random.sample(able_cells, 1)[0]
            # print('random: ', row, ", ",col)
            return row, col

        victorize_state = np.reshape(state, [1,20,20,1]) 
        prediction = self.model.predict(victorize_state)[0]

        row, col = probability_positions(state, prediction)
        print('DQN: ', row, ", ",col)
        return row, col


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))



    def replay(self, batch_size):
        r = random.sample([5,10], 1)[0]
        print('random: ', r) 
        for i in range(r):
            minibatch = random.sample(self.memory, batch_size)
            X = []
            y = []
            for state, act, reward, next_state, done in minibatch:
                X.append(state)
                state = np.reshape(state, [1,20,20,1])
                next_state = np.reshape(next_state, [1,20,20,1])
                # Tính Q-value cho trạng thái hiện tại
                target = self.model.predict(state, verbose=0)[0]
                position = act[0] * 20 + act[1]
                target[position] = reward
                if not done:
                    next_q_values = self.target_model.predict(next_state, verbose=0)[0]
                    next_max_q_value = np.max(next_q_values)
                    target[position] = reward + self.gamma * next_max_q_value
                q_values = self.model.predict(state, verbose=0)[0]
                q_values[position] = target[position]
                y.append(q_values)

            X = np.array(X)
            y =np.array(y)
                # Huấn luyện mô hình với batch hiện tại và lưu trữ giá trị loss
            self.model.fit(X, y, batch_size=batch_size, epochs=2, verbose=1)

            self.gobal_step += 1
            print('i range: ', i)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                print('cap nhat epsilon: ', self.epsilon)
            if self.gobal_step >= 5:
                self.update_target_model()
                self.gobal_step = 0
                self.num_update_target_model +=1
                print('Update weight target model')

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def save(self, name, name_weight):
        # self.model.save_weights(name)
        self.target_model.save(name)
        self.target_model.save_weights(name_weight)

    def load(self, name):
        self.model.load_weights(name)
        self.target_model.load_weights(name)
agent = DQNAgent((BOARD_SIZE, BOARD_SIZE, 1),ALPHA, GAMMA, EPSILON, EPSILON_MIN, EPSILON_DECAY, BOARD_SIZE ** 2, deque(maxlen=2000))
agent.load('models/model_DQN_V14_Weights.h5')
DQN_TURN = caro.PLAYER(-1, False)       # 0
MINIMAX_TURN = caro.PLAYER(1, True)     # X

def train(episodes, batch_size, model_file, weight_file):
    DQN_win = 0
    MINIMAX_win = 0
    range_state = 0

    for episode in range(episodes):
        board = caro.init_chess(20,5)
        done = False
        flag = False
        MINIMAX_TURN.state = True
        DQN_TURN.state = False
        while True:
            cur_state = board
            if DQN_TURN.state:
                row, col = agent.action(board)
                board[row][col] = DQN_TURN.chess
            else:
                row, col = caro.set_move_bot(board, AI = MINIMAX_TURN.chess, person = DQN_TURN.chess)
                board[row][col] = MINIMAX_TURN.chess

            DQN_TURN.set_state()
            MINIMAX_TURN.set_state()
	    
            next_state = board
            # reward = evaluate(next_state, DQN_TURN.chess, MINIMAX_TURN.chess)
            reward = rewards(next_state, DQN_TURN.chess)
            if caro.game_over(next_state, DQN_TURN.chess, MINIMAX_TURN.chess):
                if caro.check_win(next_state, DQN_TURN.chess):
                    print('DQN win ne')
                    DQN_win +=1
                if caro.check_win(next_state, MINIMAX_TURN.chess):
                    # print('DQN win ne')
                    MINIMAX_win +=1
                done = True
                flag = True
            agent.remember(cur_state, (row, col), reward, next_state, done)
            range_state +=1
            if len(agent.memory) >= 500:
                  agent.replay(batch_size)
                  agent.memory.clear()
                  range_state = 0
            # caro.show_chess_board(board)
            if flag:
                with open('logs/num_wins_v13.txt', mode='w') as f:
                    f.write("MINIMAX_TURN wins: "+ str(MINIMAX_win) + "\nDQN_TURN wins: "+ str(DQN_win))
                break
        episode+=1
        print('range_state: ', range_state)
        print('episode', episode)
        agent.save(model_file, weight_file)
    return DQN_win
DQN_win = train(EPOSIDES, BATCH_SIZE, 'models/model_DQN_V14.h5', 'models/model_DQN_V14_Weights.h5')
# print(random.randrange(5, 15))

# print('DQN_win', DQN_win, agent.num_update_target_model)

