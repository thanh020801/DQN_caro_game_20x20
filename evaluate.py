# .\myvenv\Scripts\activate
import math
import numpy as np
import random
from math import inf as infinity
# import evaluate as eval
# DEFINE BOARD 
BOARD = 20
WIN_STATE = 5
WIN = False

VALUE_SCORE_ATTACK_OPPONENT = {
	0: 0, 
	1: 11, 
	2: 110, 
	3: 10000, 
	4: 110000, 
	5: 11000000,
}
VALUE_SCORE_ATTACK_PLAYER = {
	0: 0, 
	1: 11, 
	2: 110, 
	3: 10000, 
	4: 210000, 
	5: 21000000,
}
VALUE_SCORE_DEFENSE = {
	0: 0, 
	1: 50, 
	2: 300, 
	3: 30000, 
	4: 500000, 
	5: 90000000,
}

# Tính điểm cho khoảng cách từ quân cờ đến vị trí trung tâm
def score_distance(board, player):
	n = ((BOARD-1) / 2) + 1 
	score = 0
	for i in range(BOARD):
		for j in range(BOARD):
			if board[i][j] == player:
				_,d = math.modf(math.sqrt((n - i)**2 + (n - j)**2))
				score+= 100/ (d+1) + 100 / (d+1)
			if board[i][j] == -player:
				_,d = math.modf(math.sqrt((n - i)**2 + (n - j)**2))
				score-= 100/ (d+1) + 100 / (d+1)

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
	for i in range(len(arr)):
		# Nếu cờ đối thủ ở vị trí đầu tiên
		if arr[i] == -player:
			count1+=1
			if count1 == 1:
				start = i
		#  Nếu cở đối thử ở vị trí cuối cùng
		if len(arr)-1 == i:
			if start >=0:
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
def count_evaluate_attack(arr, player, VALUE_SCORE):
	score_player = 0
	# trả lại danh sách điểm đầu và cuối liên tiếp của quân ta
	list_attack = get_list_start_end(arr, -player)
	# print(list_attack)
	for start , end in list_attack:
		block = check_block(arr, start, end, -player)
		if block == 2:
			score_player += 0
		else:
			num = end - start +1
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
		score += count_evaluate_defense(i, player)
		score += count_evaluate_attack(i, player, VALUE_SCORE_ATTACK_PLAYER)  # 11  0
		score -= count_evaluate_attack(i, opponent, VALUE_SCORE_ATTACK_OPPONENT)  # 0  -11

	for i in range(BOARD):
		temp = [row[i] for row in board]
		a=1
		score += count_evaluate_defense(temp, player)
		score += count_evaluate_attack(temp, player, VALUE_SCORE_ATTACK_PLAYER)  # 11  0
		score -= count_evaluate_attack(temp, opponent, VALUE_SCORE_ATTACK_OPPONENT)  # 0 -11

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
			score += count_evaluate_attack(i, player, VALUE_SCORE_ATTACK_PLAYER)  # 0
			score -= count_evaluate_attack(i, opponent, VALUE_SCORE_ATTACK_OPPONENT) #-11


	# print('point bot diag: ', score)
	score += score_distance(board, player)
	# print('score_distance(board, BOT.chess)', score_distance(board, BOT.chess))
	# print('score end: ', score)
	return score
# ////////////////////////////////////////////////////////////////////////////////
