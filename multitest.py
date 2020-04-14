from multiprocessing import Pool, Process, Queue
from threading import Thread, Lock, RLock
import queue
import numpy as np
import time
import random
from grid import Grid
from keras.models import Sequential, clone_model, load_model
from keras.layers import Dense, Reshape
from keras.optimizers import Adam
import tensorflow as tf

BATCH_SIZE = 32
MODEL_PATH = "models/64d-64d-32d-16d/1586434040/v003000"
PROCESSES = 4


class Simulator(Process):

	def __init__(self, samples, tasks, outs, *args, **kwargs):
		super().__init__()
		self.samples = samples
		self.tasks = tasks
		self.outs = outs

	def run(self):

		threads = [Thread(target=simulate, args=[self.samples[i], self.outs]) for i in range(*self.tasks)]

		for thread in threads:
			thread.start()

		for thread in threads:
			thread.join()

		for i in range(*self.tasks):
			self.outs.put((self.samples[i], np.random.rand(7)))


def simulate(sample, outs=None):
	time.sleep(0.01) # Compute stuff

	q_values = np.random.rand(7)

	if outs == None:
		return (sample, q_values)
	else:
		outs.put((sample, q_values))


def thread(samples):
	results = []

	outs = queue.Queue()

	threads = [Thread(target=simulate, args=[samples[i], outs]) for i in range(BATCH_SIZE)]

	for thread in threads:
		thread.start()

	for thread in threads:
		thread.join()

	for i in range(BATCH_SIZE):
		results.append(outs.get())

	return results


def pool(samples):
	with Pool(processes=PROCESSES) as pool:
		results = pool.map(simulate, samples)
	pool.join()

	return samples


def combined(samples):
	results = []

	outs = Queue()

	task = BATCH_SIZE / PROCESSES
	if int(task) != task:
		raise Exception("BATCH_SIZE must be divisible by PROCESSES!")

	pool = [Simulator(samples, (int(i*task), int((i+1)*task)), outs) for i in range(PROCESSES)]

	for process in pool:
		process.start()

	for process in pool:
		process.join()

	for i in range(BATCH_SIZE):
		results.append(outs.get())

	return results


def gpu_vs_cpu():
	a = np.zeros((BATCH_SIZE,))
	b = np.random.rand(BATCH_SIZE, 7, 6, 1)

	model = Sequential()

	model.add(Reshape((42,), input_shape=(7, 6, 1)))
	model.add(Dense(64, activation="relu"))
	model.add(Dense(64, activation="relu"))
	model.add(Dense(7, activation="relu"))

	model.compile(Adam(), loss="mse")

	print(model.predict(np.array([a[i] * b[i] for i in range(BATCH_SIZE)])))

	test = np.array([np.random.rand(7, 6, 1) for _ in range(4)])
	model.predict(test)

	start = time.time()
	with tf.device('/gpu:0'):
		model.predict(test)
	print(f"GPU : {time.time() - start}")

	start = time.time()
	with tf.device('/cpu:0'):
		model.predict(test)
	print(f"CPU : {time.time() - start}")


if __name__ == "__main__":
	

	samples = [(np.random.rand(7), random.random()) for _ in range(BATCH_SIZE)]

	print("Perfromence")

	start = time.time()
	res = thread(samples)
	end = time.time()
	print(f"Thread : {round(end - start, 3)}")

	start = time.time()
	res = pool(samples)
	end = time.time()
	print(f"Pool : {round(end - start, 3)}")

	# start = time.time()
	# res = combined(samples)
	# end = time.time()
	# print(f"Combined : {round(end - start, 3)}")