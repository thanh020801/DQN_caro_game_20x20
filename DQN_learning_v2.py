# Use target network
import tensorflow as tf
from keras.layers import Dense, Flatten, Conv2D
from keras import Input
from keras.models import Sequential, load_model
import random
import numpy as np 
from collections import deque

import caro_part4 as caro
EPOSIDES = 1000
ALPHA = 0.001
GAMMA = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.005

BOARD_SIZE = 20
BATCH_SIZE = 32



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
        self.model = self.built_model()
        self.target_model = self.built_model()

    def built_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3,3), strides=2, activation = 'relu', input_shape = self.input_shape))
        model.add(Conv2D(64, kernel_size=(3,3), strides=1, activation = 'relu'))
        model.add(Conv2D(64, kernel_size=(3,3), strides=1, activation = 'relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.num_actions, activation = 'softmax'))
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model


    def action(self, state):
        r = random.random()
        if r <= self.epsilon:
            able_cells = caro.empty_cells_around(state, 2)
            if len(able_cells) > 0:
                row, col = random.sample(able_cells, 1)[0]
            else:
                able_cells = caro.empty_cells(state)
                row, col = random.sample(able_cells, 1)[0]
            # print('DQN du doan: ', row, ", ",col)
            return row, col

        # empty_cells = empty_cells_around(state, 1)
        victorize_state = np.reshape(state, [1,20,20,1]) 
        prediction = self.model.predict(victorize_state)[0]
        # prediction = np.argmax(prediction[0])

        row, col = probability_positions(state, prediction)
        print('DQN du doan: ', row, ", ",col)
        return row, col


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))



    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, act, reward, next_state, done in minibatch:
            state = np.reshape(state, [1,20,20,1])
            next_state = np.reshape(next_state, [1,20,20,1])
            # Tính Q-value cho trạng thái hiện tại
            target = self.model.predict(state)[0]
            position = act[0] * 20 + act[1]
            target[position] = reward
            if not done:
                next_q_values = self.target_model.predict(next_state)[0]
                next_max_q_value = np.max(next_q_values)
                target[position] = reward + self.gamma * next_max_q_value
            q_values = self.model.predict(state)[0]
            q_values[position] = target[position]
            # print('q_values: ', q_values)
            # print('reshape q_value: ', q_values.reshape(-1,400))
            # Huấn luyện mô hình với batch hiện tại và lưu trữ giá trị loss
            self.model.fit(state, q_values.reshape(-1,400), epochs=1, verbose=0)
        self.gobal_step += 1
        print('gobal_step: ', self.gobal_step)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            print('cap nhat epsilon: ', self.epsilon)
        if self.gobal_step >= 32:
            self.update_target_model()
            self.gobal_step = 0
            print('Update weight target model')

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def save(self, name, name_weight):
        # self.model.save_weights(name)
        self.target_model.save(name)
        self.target_model.save_weights(name_weight)


agent = DQNAgent((BOARD_SIZE, BOARD_SIZE, 1),ALPHA, GAMMA, EPSILON, EPSILON_MIN, EPSILON_DECAY, BOARD_SIZE ** 2, deque(maxlen=2000))

board = np.zeros((20,20), dtype='int')
new_board1 = np.copy(board)
new_board1[5][5] = 1
new_board2 = np.copy(board)
new_board2[5][5] = -1
new_board3 = np.copy(board)
new_board3[5][5] = 1
# a = agent.action(board)
# print(a)
HUMAN = caro.HUMAN
BOT = caro.BOT

def train(episodes, batch_size, model_file, weight_file):
    DQN_win = 0
    

    for episode in range(episodes):
        board = caro.init_chess(20,5)
        done = False
        flag = False
        BOT.state = False
        HUMAN.state = True
        while True:
            
            if BOT.state:
                row, col = agent.action(board)
                board[row][col] = BOT.chess
            else:
                row, col = caro.set_move_bot(board)
                board[row][col] = HUMAN.chess

            HUMAN.set_state()
            BOT.set_state()
	    
            next_state = board
            reward = caro.evaluate(board)
            if caro.game_over(board, HUMAN, BOT):
                if caro.check_win(board, HUMAN):
                    print('DQN win ne')
                    DQN_win +=1
                done = True
                flag = True
            agent.remember(board, (row, col), reward, next_state, done)
            
            if len(agent.memory) >= batch_size:
                  agent.replay(batch_size)
                  agent.memory.clear()
            caro.show_chess_board(board)
            if flag:
                break
        episode+=1
        print('episode', episode)
        agent.save(model_file, weight_file)
    return DQN_win
DQN_win = train(EPOSIDES, BATCH_SIZE, 'models/model_DQN_V8.h5', 'models/model_DQN_V8_Weights.h5')


print('DQN_win', DQN_win)

# board = caro.init_chess(20, 5)
# board[5][7] = 1
# board[5][6] = -1
# board[8][8] = 1
# board[13][3] = 1
# model = load_model('models/model_DQN_v3.h5')

# victorize_state = np.array(board).reshape(1, BOARD_SIZE ** 2) 

# pred = model.predict(victorize_state)[0]
# print(pred)




# print(probability_positions(board, pred))
# sort_pred = np.sort(pred)[::-1]
# print(sort_pred)
# for i in sort_pred:
