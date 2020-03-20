import numpy as np
import pygame

RENDER = True

# Some game constants
SPACING = 5
DIAMETER = 100
SIZE = (7*(SPACING+DIAMETER)+SPACING, 6*(SPACING+DIAMETER)+SPACING)

ENCODE_PLAYER = [1, -1] # The first element of ENCODE_PLAYER is attributed to the first player etc.

class Coin(pygame.sprite.Sprite):

	def __init__(self, position, image, unique_id):
		if not pygame.get_init():
			pygame.init() # Initialize if needed

		pygame.sprite.Sprite.__init__(self) # Initialize parent class

		self.image = image
		self.rect = image.get_rect().move(position)
		self.position = position
		self.id = unique_id

	def update(self, image, unique_id):
		if self.id == unique_id:
			self.image = image
			self.rect = image.get_rect().move(self.position)

class Grid:

	def __init__(self):
		self.grid = np.zeros((7,6)) # Initialize grid with zeros
		self.coin_played = 0 # Keeps track of how many coins are already played

	def play_coin(self, value, column):
		if self.grid[column][0] != 0: # No more free space in that column
			return None
		else:
			for row in range(5, -1, -1):
				if self.grid[column][row] == 0:
					self.grid[column][row] = value # Assign given value
					self.coin_played += 1 # Increment the played coin counter
					return (column, row)

	def is_winning_coin(self, cell, value):
		if self.coin_played < 7: # It's impossible that some one has already won
			return False

		# Horizontal - 
		counter = 0
		for column in range(0, 7):
			if self.grid[column][cell[1]] == value:
				counter += 1
			else:
				counter = 0

			if counter == 4:
				return True

		# Vertical |
		counter = 0
		for row in range(0, 6):
			if self.grid[cell[0]][row] == value:
				counter += 1
			else:
				counter = 0

			if counter == 4:
				return True

		# Diagonal \
		counter = 0
		for i in range(-min(cell), min(6 - cell[0], 5 - cell[1]) + 1):
			if self.grid[cell[0]+i][cell[1]+i] == value:
				counter += 1
			else:
				counter = 0

			if counter == 4:
				return True

		# Diagonal /
		counter = 0
		for i in range(-min(cell[0], 5 - cell[1]), min(6 - cell[0], cell[1]) + 1):
			if self.grid[cell[0]+i][cell[1]-i] == value:
				counter += 1
			else:
				counter = 0

			if counter == 4:
				return True

		return False

	def is_full(self):
		return self.coin_played == 42 # 7*6

class Player:

	def __init__(self):
		if not pygame.get_init():
			pygame.init() # Initialize if needed

		self.clock = pygame.time.Clock()

	def play(self, *argv, **kwargs):
		while True and RENDER:
			self.clock.tick(30) # Show at most 30 FPS

			# Handle events
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					exit()
				if event.type == pygame.MOUSEBUTTONDOWN:
					return int((event.pos[0] - SPACING) / (SPACING + DIAMETER))

class PlayerManager:

	def __init__(self, *args, starting_player=0):
		self.players = []
		self.current_player = starting_player # 0 or 1
		self.index = starting_player # 0 or 1

		for arg in args:
			if isinstance(arg, Player) or isinstance(arg, Bot):
				self.players.append(arg)

		self.one_player = len(self.players) == 1
	
	def play(self):
		if not self.one_player:
			self.index = not self.index
		
		self.current_player = not self.current_player # 0 becomes 1 and vice versa

		cell = self.players[self.index].play()

		return self.current_player, cell

class ConnectFourGame:

	def __init__(self):
		self.over = False
		self.winner = None
		self.grid = Grid()

		if RENDER:
			if pygame.get_init():
				pygame.init() # Initialize if needed

			self.clock = pygame.time.Clock()

			self.images = [ # Load the images
				pygame.image.load('Sprites/RedCoin.png'),
				pygame.image.load('Sprites/YellowCoin.png'),
				pygame.image.load('Sprites/WhiteCoin.png'),
			]

			# Scale images
			for i in range(3):
				self.images[i] = pygame.transform.scale(self.images[i], (DIAMETER,DIAMETER))

			self.screen = pygame.display.set_mode(SIZE)
			pygame.display.set_caption("ConnectFourGame")
			pygame.display.set_icon(pygame.image.load('Sprites/icon.png'))

			self.board = pygame.Surface(SIZE)
			self.board.fill((0, 0, 255))

			self.slot_sprites = pygame.sprite.Group()

			for column in range(7):
				for row in range(6):
					self.slot_sprites.add(Coin(self.cell_to_position((column, row)), self.images[-1] , self.cell_to_id((column, row))))

			self.render()
	
	def cell_to_position(self, cell):
		return int(cell[0] * (SPACING + DIAMETER) + SPACING), int(cell[1] * (SPACING + DIAMETER) + SPACING)

	def cell_to_id(self, cell):
		return 10 * cell[0] + cell[1]

	def step(self, value, player, action):
		cell = self.grid.play_coin(value, action)

		if RENDER:
			self.slot_sprites.update(self.images[player], self.cell_to_id(cell))

		if cell == None:
			return

		if self.grid.is_winning_coin(cell, value):
			self.winner = value
			self.over = True
			return
		
		if self.grid.is_full():
			self.over = True
			return

	def render(self):
		self.slot_sprites.draw(self.board)
		self.screen.blit(self.board, (0,0))
		pygame.display.flip()

	def show_game_over_screen(self):
		font = pygame.font.Font(None, 128) # Get the default font

		if self.winner == None:
			message = "It's a draw!"
		else:
			message = "Player " + str(int(self.winner)) + " wins!"

		text = font.render(message, True, (255,255,255), (0,0,0))
		text_rect = text.get_rect()
		text_rect.center = (SIZE[0] / 2, SIZE[1] / 2) # Center the text

		self.screen.blit(text, text_rect) # Draw the text on the screen
		pygame.display.flip() # Update window

		while True:
			self.clock.tick(30) # Show at most 30 FPS

			# Handle events
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					exit()

# Connect Four Game Example 
if __name__ == '__main__':
	game = ConnectFourGame()

	player_manager = PlayerManager(Player())

	while not game.over:
		current_player, action = player_manager.play()

		game.step(ENCODE_PLAYER[current_player], current_player, action)
		
		if RENDER:
			game.render()

	if RENDER:
		game.show_game_over_screen()