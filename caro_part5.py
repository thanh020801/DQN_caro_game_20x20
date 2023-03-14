# .\myvenv\Scripts\activate
#  Cái ngu cần khắc phục
# Khi tạo được 4 con nhưng ở vị trí đó đã bị chặn 2 đầu nhưng nó vẫn lựa chọn đánh vào thay vì chặn đối thủ
import math
import numpy as np
import random
from math import inf as infinity
# import evaluate as eval
# DEFINE BOARD 
BOARD = 20
WIN_STATE = 5
WIN = False

VALUE_SCORE_ATTACK = {
	0: 0, 
	1: 11, 
	2: 110, 
	3: 10000, 
	4: 110000, 
	5: 11000000,
}
VALUE_SCORE_ATTACK1 = {
	0: 0, 
	1: 11, 
	2: 110, 
	3: 10000, 
	4: 210000, 
	5: 91000000,
}
VALUE_SCORE_DEFENSE = {
	0: 0, 
	1: 50, 
	2: 300, 
	3: 30000, 
	4: 500000, 
	5: 30000000,
}

def init_chess(n, win_state):
	i = 0
	rows, cols = (n, n)
	return [[0 for x in range(cols)] for y in range(rows)]


def show_chess_board(board):
	board = np.array(board)
	for i in range(BOARD):
		if i == 0:
			print('   ',end=' ')
		print(i, end='  ')
	print('\n')
	w = 0
	for i in board:
		print(w, ' ', end=' ')
		for j in i:
			if(j == 0):
				print('.', end='  ')
			elif j == 1:
				print('X', end='  ')
			elif j == -1:
				print('0', end='  ')
		print(' ')
		w+=1
	print('-------------------------')	


def is_empty_cell(board, row, col):
	if board[row][col] == 0:
		return True
	return False

def empty_cells(board):
	emp_cells = []
	for i in range(BOARD):
		for j in range(BOARD):
			if is_empty_cell(board, i, j):
				emp_cells.append([i,j])
	return emp_cells

def get_surrounding_positions(board, row_index, col_index, n=1):
	surrounding_positions = []

	for i in range(row_index-n, row_index+n+1):
		for j in range(col_index-n, col_index+n+1):
			if 0 <= i < len(board) and 0 <= j < len(board[0]) and (i, j) != (row_index, col_index):
				if is_empty_cell(board, i, j):
					surrounding_positions.append([i, j])

	return surrounding_positions
def empty_cells_around(board, n):
	emp_cells = []
	for i in range(BOARD):
		for j in range(BOARD):
			if not is_empty_cell(board, i,j):
				emp_cells.append(get_surrounding_positions(board, i, j , n))
	new_emp_cells = []
	for sub_arr in emp_cells:
		new_emp_cells.extend(sub_arr)
	new_emp_cells = list(set(map(tuple, new_emp_cells)))
	result = []
	for item in new_emp_cells:
		result.append(list(item))
	return result

def count_continuity(arr, player):
	count1 = 0
	max_count_user1  = 0

	for i in range(len(arr)):
		if arr[i] == player:
			count1+=1
			if count1 > max_count_user1: 
				max_count_user1 = count1
		else:
			count1 = 0
	if max_count_user1 >= WIN_STATE:
		return True
	return False


def check_win( board, player):
	# check row
	for i in board:
		check_row = count_continuity(i,player)
		if(check_row != 0):
			# print('win row', check_row)
			# WIN = True
			return True

	# check col
	for i in range(BOARD):
		temp = [row[i] for row in board]
		check_col = count_continuity(temp,player)
		if(check_col != 0):
			# print('win col', check_col)
			# WIN = True
			return True

	#  check diags
	board = np.array(board)
	diags = [board[::-1,:].diagonal(i) for i in range(-(BOARD-1), BOARD)]
	diags.extend(board.diagonal(i) for i in range(BOARD-1,-BOARD,-1))
	listDiags = [n.tolist() for n in diags]
	for i in listDiags:
		check_diags = count_continuity(i,player)
		if(check_diags != 0):
			# print('win diags', check_diags)
			# WIN = True
			return True
	# WIN = False
	if len(empty_cells(board)) == 0:
		return True

	return False



def game_over(board, player1, layer2):
	return check_win(board, player1) or check_win(board, layer2)

def action(board, player1, player2):
	row = col = None
	if player1.state:
		row, col = set_move_human()
		while not is_empty_cell(board, row, col):
			print('Vi tri khong hop le. vui long danh lai')
			row, col = set_move_human()
		board[row][col] = player1.chess
		
	elif player2.state:
		row, col = set_move_bot(board, player2.chess, player1.chess)
		board[row][col] = player2.chess
	player1.set_state()
	player2.set_state()









# DEFINE PLAYER
class PLAYER:
	def __init__(self, chess, state):
		self.chess = chess
		self.state = state

	# Update state after set_move
	def set_state(self):
		self.state = not self.state

HUMAN = PLAYER(1, True)
BOT = PLAYER(-1, False)

# Tính điểm cho khoảng cách từ quân cờ đến vị trí trung tâm
def score_distance(board, player):
	n = ((BOARD-1) / 2) + 1 
	score = 0
	for i in range(BOARD):
		for j in range(BOARD):
			if board[i][j] == player:
				_,d = math.modf(math.sqrt((n - i)**2 + (n - j)**2))
				score+= 25/ (d+1) + 25 / (d+1)
			if board[i][j] == -player:
				_,d = math.modf(math.sqrt((n - i)**2 + (n - j)**2))
				score-= 25/ (d+1) + 25 / (d+1)

	return score

# kiểm tra xem một chuổi các quân cờ của địch đã được chặn hay chưa
# nếu chặn 1 đầu thì return 1, chăn 2 đầu return 2, return 0
def check_block(arr, start, end, player):
	block = 0
	if start > 0 and end < len(arr) -1:
		if arr[start-1] == player:
			block+=1
		if arr[end + 1] == player:
			block+=1
		return block
	
	elif start == 0 and end < len(arr) -1:
		if arr[end + 1] == player:
			return 2
	elif start > 0 and end == len(arr) -1:
		if arr[start - 1] == player:
			return 2

	return 0


# Đếm số lượng cờ liên tiếp của player 
# trả về 1 danh sách các giá trị đầu và cuối của chuổi liên tiếp
def get_list_start_end(arr, player):
	count1 = 0
	list_max_count1 = []
	start = -1
	# [0,0,0,0,1,0,0,0,0,1,0,0,1]
	for i in range(len(arr)):
		# Nếu cờ đối thủ ở vị trí đầu tiên
		if arr[i] == -player:
			count1+=1
			if count1 == 1:
				start = i
		#  Nếu cở đối thử ở vị trí cuối cùng
		if len(arr)-1 == i:
			if start >=0 and arr[i] == -player:
				list_max_count1.append([start, i])
				break
		#  Nếu không phải cờ đối thủ
		if arr[i] != -player:
			count1 = 0
			if start >=0:
				list_max_count1.append([start, i-1])
			start = -1
	return list_max_count1


# Hàm tính điểm cho tấn công
def count_evaluate_attack(arr, player, VALUE_SCORE = VALUE_SCORE_ATTACK):
	score_player = 0
	# trả lại danh sách điểm đầu và cuối liên tiếp của quân ta
	list_attack = get_list_start_end(arr, -player)
	# print(list_attack)
	for start , end in list_attack:
		block = check_block(arr, start, end, -player)
		num = end - start +1
		if block == 2 and num < 5:
			score_player += 0
		else:
			if num > 5:
				num = 5
			score_player += VALUE_SCORE[num]
	return score_player

# Hàm tính điểm cho phòng thủ
def count_evaluate_defense(arr, player):
	score_player = 0
	# trả lại danh sách điểm đầu và cuối liên tiếp của quân địch
	list_defense = get_list_start_end(arr, player)
	# print('list_defense', list_defense)
	for start, end in list_defense:
		block = check_block(arr, start, end, player)
		num = end - start + 1

		if num > 5:
			num = 5
		if num == 3 and block == 1:
			score_player +=  VALUE_SCORE_DEFENSE[num] * block * 2
			continue
		elif num == 3 and block == 2:
			score_player +=  VALUE_SCORE_DEFENSE[num] * block
			continue
		score_player +=  VALUE_SCORE_DEFENSE[num] * block
	# print(score_player)
	return score_player

# Hàm tính điểm cho trạng thái hiện tại 
# Tăng điểm phòng thủ và tấn công của quân ta trừ điểm tấn công của quân địch
# Công thêm điểm khoảng cách so với vị trí trung tâm
def evaluate(board,player, opponent):
	score = 0
	for i in board:
		a=0
		score += count_evaluate_defense(i, player)
		score += count_evaluate_attack(i, player, VALUE_SCORE_ATTACK1)  # 11  0
		# print('score row', score)
		score -= count_evaluate_attack(i, opponent)  # 0  -11

	for i in range(BOARD):
		temp = [row[i] for row in board]
		a=1
		score += count_evaluate_defense(temp, player)
		score += count_evaluate_attack(temp, player, VALUE_SCORE_ATTACK1)  # 11  0
		score -= count_evaluate_attack(temp, opponent)  # 0 -11

	# điểm cho đường chéo chính và đường chéo phụ
	temp_board = np.array(board)
	diags = [temp_board[::-1,:].diagonal(i) for i in range(-(BOARD-1), BOARD)]
	diags.extend(temp_board.diagonal(i) for i in range(BOARD-1,-BOARD,-1))
	listDiags = [n.tolist()  for n in diags]
	# print(listDiags)
	for i in listDiags:
		if len(i) > 5:
			a=1
			score += count_evaluate_defense(i,  player)
			score += count_evaluate_attack(i, player, VALUE_SCORE_ATTACK1)  # 0
			# print('score row', score)
			score -= count_evaluate_attack(i, opponent) #-11


	score += score_distance(board, player)

	return score
# ////////////////////////////////////////////////////////////////////////////////


def minimax(board, depth, maximizingPlayer, alpha, beta, player, opponent):

    # Kiểm tra nếu đạt độ sâu tối đa hoặc đã có kết quả trận đấu
	if depth == 0 or game_over(board, player, opponent):
		return evaluate(board, player, opponent), [-1, -1]

    # Nếu là lượt của người chơi tối đa
	if maximizingPlayer:
		best_value = -infinity
		best_move = [-1, -1]
		for move in empty_cells(board):
			row , col = move[0], move[1] 
			board[row][col] = player
			value, m = minimax(board, depth - 1, False, alpha, beta, player, opponent)
			board[row][col] = 0
			if value > best_value:
							best_value = value
							best_move = move
							# print('best_move max', best_move)
			alpha = max(alpha, best_value)
			if beta <= alpha:
				break
		return best_value, best_move

    # Nếu là lượt của người chơi tối thiểu
	else:
		best_value = infinity
		best_move = [-1, -1]
		for move in empty_cells(board):
			row , col = move[0], move[1] 
			board[row][col] = opponent
			value, _ = minimax(board, depth-1, True, alpha, beta, player, opponent)
			board[row][col] = 0
			if value < best_value:
				best_value = value
				best_move = move
				# print('best_move min',  best_move)
			beta = min(beta, best_value)
			if beta <= alpha:
				break
		return best_value, best_move


def set_move_human():
	row = int(input('input row between 0 and 10: '))
	col = int(input('input col between 0 and 10: '))
	if row >= 0 and row < BOARD and col >=0 and col < BOARD:
		return row, col
	else:
		print('Vi Tri ko hop le')
		return set_move_human()

def set_move_bot(board, AI, person):
		row = random.randint(0, BOARD-1)
		col = random.randint(0,BOARD-1)
		if len(empty_cells_around(board, 1)) <= 0 :
			return row, col, evaluate(board, AI, person)
		
		alpha = -infinity
		beta = infinity
		cur_board = board
		
		best,move = minimax(cur_board, 1, True, alpha, beta, AI, person )
		# print('Minimax: ', best ,move)
		return move[0], move[1], best
# AI

BOARD_STATE =  init_chess(BOARD, 5)
def game():
	while True:
		show_chess_board(BOARD_STATE)
		action(BOARD_STATE, HUMAN, BOT)
		if game_over(BOARD_STATE, HUMAN.chess, BOT.chess):
			print('Ket thuc van')
			show_chess_board(BOARD_STATE)
			break
# game()
# a = [1,-1,1,-1,0,-1,1,0,-1,-1]
# print(is_blocked(a,4,1))

b = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1], 
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

# print(evaluate(b, BOT.chess, HUMAN.chess))
