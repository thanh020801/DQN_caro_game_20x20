import numpy as np
import caro_part3 as caro
# board_size = 20
alpha = 0.1
epsilon = 1
gamma = 0.6
num_episodes = 100000

BOARD_STATE =  caro.init_chess(caro.BOARD, 5)

"""
    Tạo ra bảng Q có [state, action]
    bảng đầu tiên dùng để lưu trạng thái của bàn cờ hiện tại [0]=>dòng, [1]=> cột
    Bảng thứ 2 dùng để lưu hành động cho bước tiếp theo của người chơi và đối thủ, [0]=> player, [1]=> dòng, [2]=> cột
"""
def create_q_table(board_size):
    action = np.zeros((2,20,20),dtype='int')
    # state = np.zeros((20,20),dtype='int')
    Q = []
    Q.append(BOARD_STATE)
    Q.append(action)
    return   Q  #np.zeros((20,20,2,20,20),dtype='float32')

Q_TABLE = create_q_table(20)
# print(Q_TABLE[0])
# Công thức q_learning
# Q(state, action) = (1- alpha) * Q(state, action) + alpha * (reward + gamma * max(Q(next_state, all_actions)))

Q_TABLE[1][0][11][12] = 20
# def get_next_action(state):
#     q_values = np.argmax(Q_TABLE[1][0], axis=1)
#     print(q_values)

# get_next_action(BOARD_STATE)

# def get_position_best_value(state,Q):
#     state_q = Q[0][state[0]][state[1]]
#     best_actions = np.argwhere(state_q == np.amax(state_q)).flatten()
#     print(best_actions)
#     return np.random.choice(best_actions)
# a = get_position_best_value(BOARD_STATE, Q_TABLE)
# print(a)


def get_best_action(Q, state):
    # convert state list into a numpy array
    state = np.array(state)
    
    # Tìm tất cả các ô trống trong bàn cờ
    empty_spaces = np.where(state == -1)
    
    # Tạo một tuple chứa các chỉ số của các ô trống
    state_indices = tuple(empty_spaces)
    
    # Chuyển đổi tuple đó thành một giá trị duy nhất
    state_idx = np.ravel_multi_index(state_indices, state.shape)
    
    # Lấy các giá trị Q liên quan đến trạng thái state
    q_vals = Q[1][0][state_idx]
    
    # Tìm hành động có giá trị Q lớn nhất và trả về vị trí của nó trong mảng
    best_action_idx = np.unravel_index(np.argmax(q_vals), q_vals.shape)
    
    # Lấy điểm số của hành động đó
    best_q_val = q_vals[best_action_idx]
    
    # Trả về vị trí và điểm số của hành động có giá trị Q lớn nhất
    return best_action_idx, best_q_val




best_action_idx, best_q_val = get_best_action(Q_TABLE, BOARD_STATE)
print("Best action:", best_action_idx)
print("Q-value:", best_q_val)