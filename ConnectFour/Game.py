import numpy as np
import pygame

RENDER = True

SPACING = 5
DIAMETER = 100
SIZE = (7*(SPACING+DIAMETER)+SPACING, 6*(SPACING+DIAMETER)+SPACING)

red_coin = None
yellow_coin = None
white_coin = None

class Slot(pygame.sprite.Sprite):

	def __init__(self, position, unique_id):
		if not pygame.get_init():
			pygame.init()

		pygame.sprite.Sprite.__init__(self)

		self.image = pygame.transform.scale(white_coin, (DIAMETER,DIAMETER))
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

	def play_coin(self, player, column):
		print(player)
		if self.grid[column][0] != 0:
			return None
		else:
			for row in range(5, -1, -1):
				if self.grid[column][row] == 0:
					self.grid[column][row] = player
					return (column, row)

	def pixel_to_grid(self, cell):
		row = int((cell[0] - SPACING) / (SPACING + DIAMETER))
		column = int((cell[1] - SPACING) / (SPACING + DIAMETER))
		return row, column

	def is_winning_coin(self, cell):
		return False

	def is_full(self):
		return False

class Player:

	def __init__(self):
		if not pygame.get_init():
			pygame.init()

		self.clock = pygame.time.Clock()

	def play(self, player, *argv, **kwargs):
		while True and RENDER:
			self.clock.tick(30)

			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					exit()
				if event.type == pygame.MOUSEBUTTONDOWN:
					return int((event.pos[0] - SPACING) / (SPACING + DIAMETER))

class Bot:
	pass

class Loop:

	def __init__(self, array):
		self.index = array[-1]
		self.array = array
		self.len = len(array)

	def __iter__(self):
		return self

	def __next__(self):
		self.index += 1

		if self.index >= self.len:
			self.index -= self.len

		return self.array[self.index]

class Game:

	def __init__(self):
		self.over = False
		self.grid = Grid()

		if RENDER:
			if pygame.get_init():
				pygame.init()

			self.load_images()

			self.screen = pygame.display.set_mode(SIZE)
			pygame.display.set_caption("ConnectFourGame")

			self.board = pygame.Surface(SIZE)
			self.board.fill((0, 0, 255))

			self.slot_sprites = pygame.sprite.Group()

			for column in range(7):
				for row in range(6):
					self.slot_sprites.add(Slot(self.cell_to_position((column, row)), self.cell_to_id((column, row))))

			self.render()

	def load_images(self):
		global red_coin, yellow_coin, white_coin

		red_coin = pygame.image.load('Sprites/RedCoin.png')
		yellow_coin = pygame.image.load('Sprites/YellowCoin.png')
		white_coin = pygame.image.load('Sprites/WhiteCoin.png')
	
	def cell_to_position(self, cell):
		return int(cell[0] * (SPACING + DIAMETER) + SPACING), int(cell[1] * (SPACING + DIAMETER) + SPACING)

	def cell_to_id(self, cell):
		return 10 * cell[0] + cell[1]

	def step(self, player, action):
		reward = -0.01
		cell = self.grid.play_coin(player, action)
		if cell == None:
			# Unauthorized action!
			return

		print(cell, self.grid.grid)
		if self.grid.is_winning_coin(cell) or self.grid.is_full():
			self.over = True

		if RENDER:
			self.slot_sprites.update(red_coin, self.cell_to_id(cell))

	def render(self):
		self.slot_sprites.draw(self.board)
		self.screen.blit(self.board, (0,0))
		pygame.display.flip()

game = Game()

player = Player()
player_manager = Loop([-1,1])
current_palyer = player_manager.index

while not game.over:
	action = player.play(current_palyer)
	game.step(current_palyer, action)
	current_palyer = next(player_manager)
	if RENDER:
		game.render()