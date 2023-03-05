# import numpy as np
# import random
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
# from tensorflow import keras
# BOARD_SIZE = 20
# NUM_ACTIONS = BOARD_SIZE ** 2
# GAMMA = 0.99
# EPSILON_START = 1.0
# EPSILON_MIN = 0.01
# EPSILON_DECAY = 0.999
# MEMORY_CAPACITY = 1000000
# BATCH_SIZE = 32
# TARGET_UPDATE_FREQUENCY = 10000
# LEARNING_RATE = 0.0001
# NUM_EPISODES = 1000000




# class DQNAgent:
#     def __init__(self, state_size, action_size):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.memory = []
#         self.gamma = 0.95   # discount rate
#         self.epsilon = 1.0  # exploration rate
#         self.epsilon_min = 0.01
#         self.epsilon_decay = 0.995
#         self.learning_rate = 0.001
#         self.model = self._build_model()

#     def _build_model(self):
#         # Neural Network to approximate Q-value function
#         model = Sequential()
#         model.add(Dense(128, input_dim=self.state_size, activation='relu'))
#         model.add(Dense(128, activation='relu'))
#         model.add(Dense(self.action_size, activation='linear'))
#         model.compile(loss='mse', optimizer='adam')
#         return model

#     def remember(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))

#     def act(self, state):
#         if np.random.rand() <= self.epsilon:
#             # Exploration: choose a random action
#             return np.random.randint(self.action_size)
#         # Exploitation: choose the best action from Q(s,a) values
#         act_values = self.model.predict(state)
#         return np.argmax(act_values[0])

#     def replay(self, batch_size):
#         minibatch = random.sample(self.memory, batch_size)
#         for state, action, reward, next_state, done in minibatch:
#             target = reward
#             if not done:
#                 # Bellman's Equation for Q value
#                 target = (reward + self.gamma *
#                           np.amax(self.model.predict(next_state)[0]))
#             # Update Q-value for given state and action
#             target_f = self.model.predict(state)
#             target_f[0][action] = target
#             # Train the Neural Network with state and target Q value
#             self.model.fit(state, target_f, epochs=1, verbose=0)
#         # Decrease exploration rate
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay

#     def load(self, name):
#         self.model.load_weights(name)

#     def save(self, name):
#         self.model.save_weights(name)




# def train(self, env, episodes=1000, batch_size=32, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, 
#           epsilon_min=0.01, save_freq=100, save_path=None):
#     scores = []
#     for episode in range(episodes):
#         state = env.reset()
#         done = False
#         score = 0
#         while not done:
#             # Chọn hành động
#             action = self.act(state, epsilon)
#             # Thực hiện hành động và nhận lại thông tin mới
#             next_state, reward, done, _ = env.step(action)
#             # Lưu trữ trạng thái và kết quả hành động
#             self.remember(state, action, reward, next_state, done)
#             # Cập nhật trạng thái hiện tại
#             state = next_state
#             # Tính toán điểm số và kết thúc nếu đạt điều kiện
#             score += reward
#             if done:
#                 break
#             # Làm mới bộ nhớ ký ức
#             self.replay(batch_size, gamma)
#             # Giảm dần epsilon
#             epsilon = max(epsilon_min, epsilon_decay * epsilon)
#         scores.append(score)
#         # In thông tin mỗi 100 episodes
#         if episode % 100 == 0:
#             print(f"Episode {episode}/{episodes} - Score: {score} - Epsilon: {epsilon:.3f}")
#         # Lưu trữ mô hình
#         if save_path and episode % save_freq == 0:
#             self.save(save_path)
#     return scores



# def test(self, env, episodes=10):
#     scores = []
#     for episode in range(episodes):
#         state = env.reset()
#         done = False
#         score = 0
#         while not done:
#             # Chọn hành động tốt nhất dựa trên mô hình
#             action = np.argmax(self.model.predict(state.reshape(1, self.state_size))[0])
#             # Thực hiện hành động và nhận lại thông tin mới
#             next_state, reward, done, _ = env.step(action)
#             # Cập nhật trạng thái hiện tại
#             state = next_state
#             # Tính toán điểm số
#             score += reward
#         scores.append(score)
#         # In thông tin mỗi 100 episodes
#         if episode % 100 == 0:
#             print(f"Episode {episode}/{episodes} - Score: {score}")
#     avg_score = np.mean(scores)
#     print(f"Average score over {episodes} episodes: {avg_score}")
#     return avg_score

































import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# tạo bảng chơi Caro với kích thước 20x20
board_size = 20
board = np.zeros((board_size, board_size), dtype=int)


# định nghĩa các hằng số cho các giá trị trong bảng
empty = 0
player1 = 1
player2 = -1

# kiểm tra xem người chơi đã chiến thắng hay chưa
def check_win(board, player):
    # kiểm tra theo hàng ngang
    for i in range(board_size):
        for j in range(board_size - 4):
            if board[i][j] == player and board[i][j+1] == player and board[i][j+2] == player and board[i][j+3] == player and board[i][j+4] == player:
                return True

    # kiểm tra theo hàng dọc
    for i in range(board_size - 4):
        for j in range(board_size):
            if board[i][j] == player and board[i+1][j] == player and board[i+2][j] == player and board[i+3][j] == player and board[i+4][j] == player:
                return True

    # kiểm tra theo đường chéo chính
    for i in range(board_size - 4):
        for j in range(board_size - 4):
            if board[i][j] == player and board[i+1][j+1] == player and board[i+2][j+2] == player and board[i+3][j+3] == player and board[i+4][j+4] == player:
                return True

    # kiểm tra theo đường chéo phụ
    for i in range(4, board_size):
        for j in range(board_size - 4):
            if board[i][j] == player and board[i-1][j+1] == player and board[i-2][j+2] == player and board[i-3][j+3] == player and board[i-4][j+4] == player:
                return True

    return False



import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.7   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        return model

    def remember(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))

    def act(self, state):
        r = random.random()
        print("r: ", r, "  e: ", self.epsilon)
        if r <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        print(np.argmax(act_values[0]))
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = np.array(random.sample(self.memory, batch_size))
        for state, action, reward, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        # self.model.save_weights(name)
        self.model.save(name)

class State:
    def __init__(self, board_size):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size))

    def vectorize(self):
        return self.board.reshape(1, self.board_size**2)

    def update(self, move, player):
        x, y = move
        self.board[x][y] = player

    def is_game_over(self, move, player):
        x, y = move
        for i in range(self.board_size):
            if self.board[x][i] != player:
                break
            if i == self.board_size - 1:
                return True

        for i in range(self.board_size):
            if self.board[i][y] != player:
                break
            if i == self.board_size - 1:
                return True

        if x == y:
            for i in range(self.board_size):
                if self.board[i][i] != player:
                    break
                if i == self.board_size - 1:
                    return True

        if x + y == self.board_size - 1:
            for i in range(self.board_size):
                if self.board[i][(self.board_size-1)-i] != player:
                    break
                if i == self.board_size - 1:
                    return True

        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 0:
                    return False
        return True


print('bat dau')
if __name__ == '__main__':
    state_size = 400   # 20 x 20
    action_size = 400   # 20 x 20
    agent = Agent(state_size, action_size)
    state = State(20)
    done = False
    batch_size = 32
    num = 0
    while not done and num < 3000:
        move = agent.act(state.vectorize())
        player = 1 if len(agent.memory) % 2 == 0 else -1
        state.update((move // 20, move % 20), player)
        
        if state.is_game_over((move // 20, move % 20), player):
            reward = 1 
            print("num ; ", num)
            num+=1
        else: 
            reward = -0.1
        done = reward == 1
        agent.remember(state.vectorize(), move, reward, done)
        state = State(20) if done else state

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
            

            agent.save('models/model_DQN_v4.h5')



# print(random.random())


# import numpy as np
# from tensorflow import keras
# import caro_part3 as caro



# model = keras.models.load_model('models/model_DQN_v3.h5')

# board = caro.init_chess(20, 5)

# # board[4][4] = -1
# # board[4][3] = 1
# # board[5][4] = -1
# # board[2][3] = 1


# # board = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, -1, -1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, -1, 1, 1, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, -1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, -1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]


# def vectorize(board):
#     return np.array(board).reshape(1, 20**2)
# print(vectorize(board))
# caro.show_chess_board(board)

# pred = model.predict(vectorize(board))
# print(pred)
# pred = np.argmax(pred,axis = 1)
# print(pred)
# row, col = divmod(pred,20)
# print(row[0],col[0])
# board[row[0]][col[0]] = 1

# caro.show_chess_board(board)

# pred = model.predict(vectorize(board))
# print(pred)
# pred = np.argmax(pred,axis = 1)
# print(pred)
# row, col = divmod(pred,20)
# print(row[0],col[0])
# board[row[0]][col[0]] = 1

# caro.show_chess_board(board)