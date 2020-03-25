import pygame

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

	def __init__(self, *args, delay=None, starting_player=0):
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