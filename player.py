import pygame

class Player:

	def __init__(self, param):
		if not pygame.get_init():
			pygame.init() # Initialize if needed

		self.clock = pygame.time.Clock()

		self.param = param

	def play(self, *argv, **kwargs):
		if self.param["RENDER"]:
			while True:
				self.clock.tick(30) # Show at most 30 FPS

				# Handle events
				for event in pygame.event.get():
					if event.type == pygame.QUIT:
						exit()
					if event.type == pygame.MOUSEBUTTONDOWN:
						return int((event.pos[0] - self.param["SPACING"]) / (self.param["SPACING"] + self.param["DIAMETER"]))
		else:
			action = -1

			while action > 6 or action < 0:
				action = int(input("Chose an action [Number between 1 - 7] : ")) - 1

			return action

class PlayerManager:

	def __init__(self, *args, starting_player=0):
		self.players = []
		self.current_player = starting_player # 0 or 1
		self.index = starting_player # 0 or 1

		for arg in args:
			if isinstance(arg, Player):
				self.players.append(arg)

		self.one_player = len(self.players) == 1
	
	def play(self, *args, **kwargs):
		if not self.one_player:
			self.index = not self.index
		
		self.current_player = not self.current_player # 0 becomes 1 and vice versa

		cell = self.players[self.index].play(*args, **kwargs)

		return self.current_player, cell