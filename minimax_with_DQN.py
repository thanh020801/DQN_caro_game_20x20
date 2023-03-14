import pygame
import caro_part5 as caro

import numpy as np
from tensorflow import keras

model = keras.models.load_model('models/model_DQN_V20.h5')
model.load_weights('models/model_DQN_V20_Weights.h5')
# tim phan tu co vi tri hop le va co xac suat du doan cao nhat
def probability_positions(board, prediction):
    # emp = caro.empty_cells_around(board,2)
    # if len(emp) <=0:
    emp = caro.empty_cells(board)
    probs = []
    for row, col in emp:
        position = row * caro.BOARD + col
        prob_position = prediction[position] 
        probs.append([row, col, prob_position])
    sort_probs = sorted(probs, key = lambda x: x[2], reverse=True)
    max_prob = sort_probs[0]
    return max_prob[0], max_prob[1]


def vectorize(board):
    return np.array(board).reshape(1, 20**2)

pygame.init()

HUMAN = caro.HUMAN
BOT = caro.BOT

BOARD = caro.BOARD
# Thiết lập màn hình
screen = pygame.display.set_mode((500, 550))
pygame.display.set_caption("Game caro 20x20")


# Màu sắc
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)

GREEN = (0, 255, 0)
RED = ( 195, 238, 155)
A = (44, 45, 46)
screen.fill(A)
# Vẽ lưới caro
block_size = 25  # kích thước mỗi ô vuông
for x in range(0, 500, block_size):
    for y in range(0, 500, block_size):
        rect = pygame.Rect(x, y, block_size, block_size)
        pygame.draw.rect(screen, RED, rect, 1)


# Vẽ quân cờ
def draw_x(x, y):
    pygame.draw.line(screen, WHITE, (x, y), (x + block_size-3, y + block_size-3), 3)
    pygame.draw.line(screen, WHITE, (x + block_size, y), (x, y + block_size), 3)

def draw_o(x, y):
    pygame.draw.circle(screen, GREEN, (x + block_size//2, y + block_size//2), block_size//2 - 3, 3)
    
# Khởi tạo ma trận 20x20 để lưu vị trí các quân cờ
board = [[0] * BOARD for _ in range(BOARD)]

# Vẽ quân cờ
def draw_board(board):
    for i in range(BOARD):
        for j in range(BOARD):
            if board[i][j] == 1:
                draw_x(j * block_size, i * block_size)
            elif board[i][j] == -1:
                draw_o(j * block_size, i * block_size)



# Xử lý sự kiện
def handle_events(board, player, opponent):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

        elif player.state:
            # row, col = caro.set_move_bot(board)
            # board[row][col] = player.chess
            # draw_board(board)
            # player.set_state()
            # opponent.set_state()

            cur_board = np.reshape(board, [1,20,20,1])
            pred = model.predict(cur_board)[0]
            # pred = np.argmax(pred,axis = 1)
            # row, col = divmod(pred,20)
            # print("row: ", row, ",col: ", col)
            row, col = probability_positions(board, pred)
            board[row][col] = player.chess
            draw_board(board)
            player.set_state()
            opponent.set_state()  

        elif opponent.state:
            # cur_board = board
            # pred = model.predict(vectorize(cur_board))
            # pred = np.argmax(pred,axis = 1)
            # row, col = divmod(pred,20)
            # print("row: ", row, ",col: ", col)
            # board[row[0]][col[0]] = opponent.chess
            # draw_board(board)
            # player.set_state()
            # opponent.set_state()
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                row, col = y // block_size, x // block_size
                if board[row][col] == 0:
                    board[row][col] = opponent.chess
                    draw_board(board)
                    player.set_state()
                    opponent.set_state()
            else:
                continue
        else:
                continue
        

# Vòng lặp chính




# while True:
#     handle_events(board, HUMAN, BOT)
#     # player = -player
#     pygame.display.update()

BOARD_STATE =  caro.init_chess(caro.BOARD, 5)


BOT.state = True
HUMAN.state = False
def game():
    while True:
        # show_chess_board(BOARD_STATE)
        handle_events(BOARD_STATE, caro.BOT, caro.HUMAN)
        pygame.display.update()
        with open('board.txt', mode='w') as f:
            f.write(str(BOARD_STATE))
        if caro.game_over(BOARD_STATE, caro.BOT.chess, caro.HUMAN.chess):
            print('Ket thuc van')
            # show_chess_board(BOARD_STATE)
            break
		

game()


path_w = 'board.txt'


with open(path_w, mode='w') as f:
    f.write(str(BOARD_STATE))
with open(path_w) as f:
    print(f.read())