import pygame


class Player:
	'''
	Note : A pygame window is necessary! Else it will raise an error.
	If you need a Player but don't want to render a pygame window use PlayerConsole

	Use : Just click the coloum (on the screen) or the number (form 1 to 7) you want to play!
	Note : If you click on a full coloum you've wasted your move and it's the other players turn!
	'''

	def __init__(self, param):
		pygame.init()
		# if not pygame.get_init():
		# 	pygame.init() # Initialize if needed
		# 	print("Initialized pygame!")

		if pygame.display.get_surface() == None:
			raise Exception("Please call pygame.display.set_mode() before initializing a Player instance! Or use a PlayerConsole!")

		self.clock = pygame.time.Clock()

		self.param = param

	def play(self, **kwargs):
		while True:
			self.clock.tick(30) # Show at most 30 FPS

			# Handle events
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					exit()
				if event.type == pygame.KEYDOWN:
					if event.key <= ord("7") and event.key >= ord("1"):
						return int(chr(event.key)) - 1
				if event.type == pygame.MOUSEBUTTONDOWN:
					return int((event.pos[0] - self.param["SPACING"]) / (self.param["SPACING"] + self.param["DIAMETER"]))

class PlayerConsole(Player):

	def __init__(self):
		pass

	def play(self, player=None, **kwargs):
		action = -1

		while action > 6 or action < 0:
			action = int(input(f"Player {player} please chose an action [Number between 1 - 7] : ")) - 1

		return action


class PlayerManager:

	def __init__(self, *args):
		self.players = []
		self.current_player = 1 # 1 or -1
		self.index = 0 # 0 or 1

		if len(args) >  2 or len(args) == 0:
			raise Exception(f"Only takes one or two args! You passed {len(args)} args.")

		for arg in args:
			if isinstance(arg, Player):
				self.players.append(arg)
			else:
				raise Exception(f"The args must be an instance of Player! You passed an object of type : {type(arg)}")

		self.two_player = len(self.players) == 2
	
	def reset(self):
		self.current_player = 1
		self.index = 0

	def play(self, **kwargs):
		action = self.players[self.index].play(**kwargs, player=self.current_player)

		if self.two_player:
			self.index = not self.index # If two different players : index changes form 0 (False) to 1 (True) and vice versa
		
		self.current_player *= -1 # 1 becomes -1 and vice versa

		return -self.current_player, action