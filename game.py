import numpy as np
import pygame
import time
from grid import Grid
from player import Player, PlayerManager

class GameParam:
	RENDER = True

	SPACING = 5
	DIAMETER = 100
	SIZE = (7*(SPACING+DIAMETER)+SPACING, 6*(SPACING+DIAMETER)+SPACING)

	ENCODE_PLAYER = [1, -1] # The first element of ENCODE_PLAYER is attributed to the first player etc.

	# Rewards
	UNAUTHORIZED = -1
	ACTION = -0.01
	WIN = 1
	DRAW = 0.5
	LOSE = -1

	def __init__(self, *args, **kwargs):
		# Setting passed values
		for key, value in kwargs.items():
			if key in locals():
				self.locals()[key] = value

# RENDER = True

# # Some game constants
# SPACING = 5
# DIAMETER = 100
# SIZE = (7*(SPACING+DIAMETER)+SPACING, 6*(SPACING+DIAMETER)+SPACING)

# ENCODE_PLAYER = [1, -1] # The first element of ENCODE_PLAYER is attributed to the first player etc.

# # Rewards
# UNAUTHORIZED = -1
# ACTION = -0.01
# WIN = 1
# DRAW = 0.5
# LOSE = -1

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
		# Updates only if the ids matches or if unique_id is None -> means every coin of the group gets updated
		if self.id == unique_id or unique_id == None:
			self.image = image
			self.rect = image.get_rect().move(self.position)

class ConnectFourGame:

	def __init__(self, param=None):
		self.over = False
		self.winner = None
		self.grid = Grid()

		if param == None:
			self.param = GameParam() # If param is None, default values will be used
		else:
			self.param = param

		if self.param.RENDER:
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
				self.images[i] = pygame.transform.scale(self.images[i], (self.param.DIAMETER,self.param.DIAMETER))

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
	
	def reset(self):
		self.over = False
		self.winner = None
		self.grid = Grid()

		if self.param.RENDER:
			self.slot_sprites.update(self.images[-1], None) # Set every coin back to empty

		return self.get_state()

	def cell_to_position(self, cell):
		return int(cell[0] * (self.param.SPACING + self.param.DIAMETER) + self.param.SPACING), int(cell[1] * (self.param.SPACING + self.param.DIAMETER) + self.param.SPACING)

	def cell_to_id(self, cell):
		return 10 * cell[0] + cell[1]

	def compute_next_state(self, state, action):
		pass

	def step(self, value, player, action):
		cell = self.grid.play_coin(value, action) # Play

		new_state = self.get_state()

		if RENDER and cell != None:
			self.slot_sprites.update(self.images[player], self.cell_to_id(cell))

		if cell == None:
			return UNAUTHORIZED, new_state

		if self.grid.is_winning_coin(cell, value):
			self.winner = value
			self.over = True
			return  WIN, new_state
		
		if self.grid.is_full():
			self.over = True
			return DRAW, new_state

		return ACTION, new_state

	def get_state(self):
		return self.grid.get_grid()

	def render(self):
		self.slot_sprites.draw(self.board)
		self.screen.blit(self.board, (0,0))
		pygame.display.flip()

	def pause(self, seconds):
		if self.param.RENDER:
			start = time.time()
			while (time.time() - start) < seconds:
				self.clock.tick(30) # Show at most 30 FPS

				# Handle events
				for event in pygame.event.get():
					if event.type == pygame.QUIT:
						exit()
		else:
			time.sleep(seconds)

	def show_game_over_screen(self):
		if self.winner == None:
			message = "It's a draw!"
		else:
			message = "Player " + str(int(self.winner)) + " wins!"
		
		if RENDER:
			font = pygame.font.Font(None, 128) # Get the default font

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
					if event.type == pygame.MOUSEBUTTONDOWN:
						return
		else:
			print(message, "\n")

# Connect Four Game Example 
if __name__ == '__main__':
	game = ConnectFourGame()

	player_manager = PlayerManager(Player())

	while not game.over:
		current_player, action = player_manager.play()

		reward, new_state = game.step(ENCODE_PLAYER[current_player], current_player, action)
		
		if RENDER:
			game.render()

	if RENDER:
		game.show_game_over_screen()