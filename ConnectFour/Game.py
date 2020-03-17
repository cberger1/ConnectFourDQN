import numpy as np
import pygame

RENDER = True

SPACING = 5
DIAMETER = 100
SIZE = (7*(SPACING+DIAMETER)+SPACING, 6*(SPACING+DIAMETER)+SPACING)

ENCODE_PLAYER = [1, -1] # The first element of ENCODE_PLAYER is attributed to the first player etc.

class Slot(pygame.sprite.Sprite):

	def __init__(self, position, image, unique_id):
		if not pygame.get_init():
			pygame.init()

		pygame.sprite.Sprite.__init__(self)

		self.image = pygame.transform.scale(image, (DIAMETER,DIAMETER))
		self.rect = self.image.get_rect().move(position)
		self.position = position
		self.id = unique_id

	def update(self, image, unique_id):
		if self.id == unique_id:
			self.set_image(image)

	def set_image(self, image):
		self.image = pygame.transform.scale(image, (DIAMETER,DIAMETER))

class Grid:

	def __init__(self):
		self.grid = np.zeros((7,6))
		self.coin_played = 0

	def play_coin(self, value, column):
		if self.grid[column][0] != 0:
			return None
		else:
			for row in range(5, -1, -1):
				if self.grid[column][row] == 0:
					self.grid[column][row] = value
					self.coin_played += 1
					return (column, row)

	def is_winning_coin(self, cell, value):
		# Horizontal
		counter = 0
		for column in range(max(0, cell[0] - 3), min(6, cell[0] + 3)):
			if self.grid[column][cell[1]] == value:
				counter += 1
			else:
				counter = 0

			if counter == 4:
				return True

		# Vertical
		counter = 0
		for row in range(max(0, cell[1] - 3), min(5, cell[1] + 3)):
			if self.grid[cell[0]][row] == value:
				counter += 1
			else:
				counter = 0

			if counter == 4:
				return True

		# Diagonal /
		start = min(3 if (cell[0]-3) >= 0 else cell[0], 3 if (cell[1]-3) >= 0 else cell[1])
		stop = min(3 if (cell[0]+3) <= 6 else 6-cell[0], 3 if (cell[1]+3) <= 5 else 5-cell[1]) + 1
		for i in range(-start, stop):
				if self.grid[cell[0]+i][cell[1]+i] == value:
					counter += 1
				else:
					counter = 0

				if counter == 4:
					return True

		# Diagonal \
		start = min(3 if (cell[0]+3) <= 6 else cell[0], 3 if (cell[1]-3) >= 0 else cell[1])
		stop = min(3 if (cell[0]-3) >= 0 else 6-cell[0], 3 if (cell[1]+3) <= 5 else 5-cell[1]) + 1
		for i in range(-start, stop):
				if self.grid[cell[0]+i][cell[1]+i] == value:
					counter += 1
				else:
					counter = 0

				if counter == 4:
					return True

		return False

	def is_full(self):
		return self.coin_played == 7*6

class Player:

	def __init__(self):
		if not pygame.get_init():
			pygame.init()

		self.clock = pygame.time.Clock()

	def play(self, *argv, **kwargs):
		while True and RENDER:
			self.clock.tick(30)

			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					exit()
				if event.type == pygame.MOUSEBUTTONDOWN:
					return int((event.pos[0] - SPACING) / (SPACING + DIAMETER))

class Bot:
	
	def __init__(self):
		pass

	def play(self):
		pass

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

class Game:

	def __init__(self):
		self.over = False
		self.grid = Grid()

		if RENDER:
			if pygame.get_init():
				pygame.init() # Initialize if needed

			self.images = [ # Load the images
				pygame.image.load('Sprites/RedCoin.png'),
				pygame.image.load('Sprites/YellowCoin.png'),
				pygame.image.load('Sprites/WhiteCoin.png'),
			]

			self.screen = pygame.display.set_mode(SIZE)
			pygame.display.set_caption("ConnectFourGame")

			self.board = pygame.Surface(SIZE)
			self.board.fill((0, 0, 255))

			self.slot_sprites = pygame.sprite.Group()

			for column in range(7):
				for row in range(6):
					self.slot_sprites.add(Slot(self.cell_to_position((column, row)), self.images[-1] , self.cell_to_id((column, row))))

			self.render()
	
	def cell_to_position(self, cell):
		return int(cell[0] * (SPACING + DIAMETER) + SPACING), int(cell[1] * (SPACING + DIAMETER) + SPACING)

	def cell_to_id(self, cell):
		return 10 * cell[0] + cell[1]

	def step(self, value, player, action):
		reward = -0.01
		cell = self.grid.play_coin(value, action)
		if cell == None:
			# Unauthorized action!
			return

		print(self.grid.grid)
		if self.grid.is_winning_coin(cell, value) or self.grid.is_full():
			self.over = True

		if RENDER:
			self.slot_sprites.update(self.images[player], self.cell_to_id(cell))

	def render(self):
		self.slot_sprites.draw(self.board)
		self.screen.blit(self.board, (0,0))
		pygame.display.flip()

game = Game()

player_manager = PlayerManager(Player())

while not game.over:
	current_player, action = player_manager.play()
	game.step(ENCODE_PLAYER[current_player], current_player, action)
	if RENDER:
		game.render()