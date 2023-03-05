import tensorflow as tf
from keras.layers import Dense, Flatten
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
EPSILON_DECAY = 0.005
EPSILON_MIN = 0.99

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
    def __init__(self, alpha, gamma, epsilon, epsilon_min, epsilon_decay, classes, memory):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.classes = classes
        self.memory = memory
        self.model = self.built_model()

    def built_model(self):
        model = Sequential()
        model.add(Input(shape=(400,)))
        # model.add(Flatten())
        model.add(Dense(256, activation = 'relu'))
        model.add(Dense(256, activation = 'relu'))
        model.add(Dense(256, activation = 'relu'))
        model.add(Dense(self.classes, activation = 'linear'))
        
        model.compile(optimizer='adam', loss='mse')
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
        victorize_state = np.array(state).reshape(1, BOARD_SIZE ** 2) 
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
            # Chuyển đổi state và next_state sang dạng ma trận (1, BOARD_SIZE ** 2)
            state_matrix = np.array(state).reshape(1, BOARD_SIZE ** 2)
            next_state_matrix = np.array(next_state).reshape(1, BOARD_SIZE ** 2)

            # Tính Q-value cho trạng thái hiện tại
            target = self.model.predict(state_matrix)[0]
            # target = target.reshape(1, BOARD_SIZE ** 2)
            # print('target 1', target)
            position = act[0] * 20 + act[1]
            # Nếu trò chơi đã kết thúc, giá trị target sẽ bằng phần thưởng
            if done:
                target[position] = reward
                # print('target 2', target)
            else:
                # Tính Q-value cho trạng thái kế tiếp
                next_q_values = self.model.predict(next_state_matrix)[0]
                next_max_q_value = np.max(next_q_values)
                # print('next_max_q_value 3', next_max_q_value)
                # Tính giá trị target dựa trên công thức Bellman
                target[position] = reward + self.gamma * next_max_q_value
                # print('target 3', target)

            # Huấn luyện mô hình với batch hiện tại và lưu trữ giá trị loss
            self.model.fit(state_matrix, target.reshape(-1,400), epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        # Trả về giá trị của mô hình sau khi đã được cập nhật
        # return self.model

    def save(self, name):
        # self.model.save_weights(name)
        self.model.save(name)


agent = DQNAgent(ALPHA, GAMMA, EPSILON, EPSILON_MIN, EPSILON_DECAY, BOARD_SIZE ** 2, deque(maxlen=2000))

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

def train(episodes, batch_size, model_file):
    DQN_win = 0
    

    for episode in range(episodes):
        board = caro.init_chess(20,5)
        done = False
        flag = False
        BOT.state = False
        HUMAN.state = True
        while True:
            caro.show_chess_board(board)
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
            
            if len(agent.memory) > batch_size:
                  agent.replay(batch_size)
                  agent.memory = []

            if flag:
                break
        episode+=1
        print('episode', episode)
        agent.save(model_file)
    return DQN_win
DQN_win = train(EPOSIDES, BATCH_SIZE, 'models/model_DQN_V6.h5')


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
