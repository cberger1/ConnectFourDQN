import numpy as np
import pygame
import time
from settings import Settings
from grid import Grid
from player import Player, PlayerManager

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
			self.set_image(image)

	def set_image(self, image):
		self.image = image
		self.rect = image.get_rect().move(self.position)

	def equal(self, unique_id):
		return self.id == unique_id


class ConnectFourGame:

	def __init__(self, param, display=True):
		self.over = False
		self.winner = None
		self.display = None
		self.grid = Grid()

		# pygame
		pygame.init()
		
		self.param = param # Game settings

		self.clock = pygame.time.Clock() # Get a referecne to the clock

		self.images = [ # Load the images
			pygame.image.load('Sprites/WhiteCoin.png'), # Empty slot
			pygame.image.load('Sprites/YellowCoin.png'), # Player 1
			pygame.image.load('Sprites/RedCoin.png'), # Player -1
		]

		# Scale images
		for i in range(3):
			self.images[i] = pygame.transform.scale(self.images[i], (self.param["DIAMETER"],self.param["DIAMETER"]))

		# Board surface (just filled wiht blue color)
		self.board = pygame.Surface(self.param["SIZE"])
		self.board.fill((0, 0, 255))

		# A sprite group to hold all coins
		self.slot_sprites = pygame.sprite.Group()

		# Add coins to the group
		for column in range(7):
			for row in range(6):
				self.slot_sprites.add(Coin(self.cell_to_position((column, row)), self.images[0] , self.cell_to_id((column, row))))

		self.set_display_mode(display) # Opens a game window if display == true
	
	def set_display_mode(self, display):
		if self.display == display: # Check if there is no need to change the mode
			return

		self.display = display

		if display:
			self.screen = pygame.display.set_mode(self.param["SIZE"])

			pygame.display.set_caption("ConnectFourGame")
			pygame.display.set_icon(pygame.image.load('Sprites/icon.png'))

			self.sync_screen()
			self.render()
		else:
			pygame.quit()

	def sync_screen(self):
		# Syncronize the screen with the grid
		for coin in self.slot_sprites.sprites():
			cell = self.id_to_cell(coin.id)
			coin.set_image(self.images[int(self.grid[cell])])

	def reset(self):
		self.over = False
		self.winner = None
		
		self.grid.clear() # Clear the gird

		self.slot_sprites.update(self.images[0], None) # Set every coin back to white

		return self.get_state()

	def cell_to_position(self, cell):
		# Calculates the position of a given cell
		x = int(cell[0] * (self.param["SPACING"] + self.param["DIAMETER"]) + self.param["SPACING"])
		y = int(cell[1] * (self.param["SPACING"] + self.param["DIAMETER"]) + self.param["SPACING"])
		return (x, y)

	def cell_to_id(self, cell):
		# Retruns the unique_id of a cell : Example (2, 4) -> 24 ; Note : (0, x) -> x
		return int(10 * cell[0] + cell[1])

	def id_to_cell(self, unique_id):
		# Transforms first the id into a string then to a cell : Example 24 -> "24" -> (2, 4)
		string = "{0:0>2.0f}".format(unique_id) # Some formatting to deal with single didigt numbers
		return (int(string[0]), int(string[1]))

	def step(self, value, player, action):
		cell = self.grid.play_coin(value, action) # Play

		new_state = self.grid.get_grid()

		if self.display and cell != None:
			self.slot_sprites.update(self.images[value], self.cell_to_id(cell))

		if cell == None:
			return self.param["UNAUTHORIZED"], new_state

		if self.grid.is_winning_coin(cell, value):
			self.winner = value
			self.over = True
			return  self.param["WIN"], new_state
		
		if self.grid.is_full():
			self.over = True
			return self.param["DRAW"], new_state

		return self.param["ACTION"], new_state

	def get_state(self):
		return self.grid.get_grid()

	def render(self):
		if not self.display:
			raise Exception("set display mode to true before rendering!")

		self.slot_sprites.draw(self.board)
		self.screen.blit(self.board, (0,0))
		pygame.display.flip()

	def pause(self, seconds):
		if self.display:
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

		if self.display:
			pygame.font.init() 

			font = pygame.font.Font(None, 128) # Get the default font

			text = font.render(message, True, (255,255,255), (0,0,0))
			text_rect = text.get_rect()
			text_rect.center = (self.param["SIZE"][0] / 2, self.param["SIZE"][1] / 2) # Center the text

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
			input(message + "\nPress ENTER to leave")


# Connect Four Game Example 
if __name__ == '__main__':
	param = Settings(RENDER=True)

	game = ConnectFourGame(param, param["RENDER"])

	player_manager = PlayerManager(Player(param))

	while not game.over:
		current_player, action = player_manager.play()

		value = param["ENCODE_PLAYER"][current_player]

		reward, new_state = game.step(value, current_player, action)
		
		if param["RENDER"]:
			game.render()
		else:
			print("Action : ", action)

	game.show_game_over_screen()