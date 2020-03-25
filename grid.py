import numpy as np

class Grid:

	def __init__(self):
		self.grid = np.zeros((7, 6, 1)) # Initialize grid with zeros
		self.coin_played = 0 # Keeps track of how many coins are already played

	def play_coin(self, value, column):
		if self.grid[column][0][0] != 0: # No more free space in that column
			return None
		else:
			for row in range(5, -1, -1):
				if self.grid[column][row][0] == 0:
					self.grid[column][row][0] = value # Assign given value
					self.coin_played += 1 # Increment the played coin counter
					return (column, row)

	def get_grid(self):
		return self.grid

	def is_winning_coin(self, cell, value):
		if self.coin_played < 7: # It's impossible that some one has already won
			return False

		# Horizontal - 
		counter = 0
		for column in range(0, 7):
			if self.grid[column][cell[1]][0] == value:
				counter += 1
			else:
				counter = 0

			if counter == 4:
				return True

		# Vertical |
		counter = 0
		for row in range(0, 6):
			if self.grid[cell[0]][row][0] == value:
				counter += 1
			else:
				counter = 0

			if counter == 4:
				return True

		# Diagonal \
		counter = 0
		for i in range(-min(cell), min(6 - cell[0], 5 - cell[1]) + 1):
			if self.grid[cell[0]+i][cell[1]+i][0] == value:
				counter += 1
			else:
				counter = 0

			if counter == 4:
				return True

		# Diagonal /
		counter = 0
		for i in range(-min(cell[0], 5 - cell[1]), min(6 - cell[0], cell[1]) + 1):
			if self.grid[cell[0]+i][cell[1]-i][0] == value:
				counter += 1
			else:
				counter = 0

			if counter == 4:
				return True

		return False

	def is_full(self):
		return self.coin_played == 42 # 7*6