
'''
Constant Container Class


All import settings are stored in this class

When creating a new Settings Object, you can pass some param you want to change
Example : param = Settings(RENDER=False, SPACING=10)

It is possible to directly index a settings Object
Example : if param["RENDER"]:

Those parameters/settings are then used to initialize a Game or Player Object
Important : Always use the same settings for the Player and the Game he is used for !
'''

class Settings:
	
	const = {
		# Render
		"RENDER" : True,
		# Board settings
		"SPACING" : 5,
		"DIAMETER" : 100,
		"SIZE" : (740 , 636),
		# Encoding
		"ENCODE_PLAYER" : [1, -1],
		# Rewards
		"UNAUTHORIZED" : -1,
		"ACTION" : -0.01,
		"WIN" : 1,
		"DRAW" : 0.5,
		"LOSE" : -1,
	}

	def __init__(self, const=None, **kwargs):
		
		if const == None:
			for key, value in kwargs.items():
				if key in self.const:
					self.const[key] = value
		else:
			self.const = const

		self.update_size()

	def __getitem__(self, key):
		return self.const[key]

	def __setitem__(self, key, value):
		self.const[key] = value

	def __contains__(self, item):
		return item in self.const

	def update_size(self):
		x = int(7 * (self.const["SPACING"] + self.const["DIAMETER"]) + self.const["SPACING"])
		y = int(6 * (self.const["SPACING"] + self.const["DIAMETER"]) + self.const["SPACING"])
		self.const["SIZE"] = (x, y)