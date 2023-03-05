import pygame
import caro_part5 as caro
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
            row, col = caro.set_move_bot(board, player.chess, opponent.chess)
            board[row][col] = player.chess
            draw_board(board)
            player.set_state()
            opponent.set_state()
        elif opponent.state:
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


a  = 0

with open('board.txt', mode='w') as f:
    f.write(str(BOARD_STATE))
with open('board.txt') as f:
    print(f.read())
    a = f.read()

