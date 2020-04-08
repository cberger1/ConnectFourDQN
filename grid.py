import numpy as np


class Grid:

	def __init__(self):
		self.grid = np.zeros((7, 6, 1)) # Initialize grid with zeros
		self.coin_played = 0 # Keeps track of how many coins are already played
		self.free_column = list(range(7)) # Keeps track of the free columns

	def __getitem__(self, key):
		return self.grid[key]

	def __setitem__(self, key, value):
		self.grid[key] = value

	def __str__(self):
		'''
		Retruns a nice visualisation of the grid
		Just use print(YOUR GRID OBJECT)
		'''

		string = "\n"

		for column in range(7):
			string += "|"

			for row in range(6):
				string += " " + str(int(self.grid[column][row][0])) + " |"

			string += "\n"

		return string

	def play_coin(self, value, column):
		if column == None:
			raise Exception("column can't equal to None!")

		if self.grid[column][0][0] != 0: # No more free space in that column
			return None
		else:
			for row in range(5, -1, -1):
				if self.grid[column][row][0] == 0:
					self.grid[column][row][0] = value # Assign given value
					self.coin_played += 1 # Increment the played coin counter

					if row == 0:
						self.free_column.remove(column) # Update the free column set

					return (column, row) # Retrun the cell where the coin was played

	def clear(self):
		self.grid = np.zeros((7, 6, 1))
		self.coin_played = 0
		self.free_column = list(range(7))

	def set_grid(self, grid):
		self.grid = grid.copy()
		self.coin_played = 0
		self.free_column = set(range(7))

		for column in range(7):
			if self.grid[column][0][0] != 0:
				self.free_column.remove(column)

			for row in range(6):
				if self.grid[column][row][0] != 0:
					self.coin_played += 1

	def get_grid(self):
		return self.grid

	def is_full(self):
		return self.coin_played == 42 # 7*6

	def is_column_full(self, state, column):
		return state[column][0][0] != 0

	def is_winning_coin(self, cell, value):
		'''
		This funciton checks if the given coin (cell is its coordinate) connects to four other coins
		Important : After playing a coin (function : play_coin()) remeber to always check if it is a winning coin!
					Or you might not see that the game is won by a player
		'''

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