# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# from multiprocessing import Pool, Process, Queue, Lock
from threading import Thread, Lock, RLock
from queue import Queue
import numpy as np
import time
import random
from keras.models import Sequential, clone_model, load_model
from keras.layers import Dense, Reshape
from keras.optimizers import Adam
import tensorflow as tf


BATCH_SIZE = 32
MODEL_PATH = "models/64d-64d-32d-16d/1586434040/v003000"
lock = Lock()

counter = 0


def prepare_sample(tasks, outs, lock):
	if tasks.empty():
		return

	# model = load_model(MODEL_PATH)

	sample = tasks.get()

	# q_values = model.predict(np.array([sample]))

	# lock.acquire()
	# try:
	# 	# print('Acquired a lock')
	# 	counter += 1
	# finally:
	# 	# print('Released a lock')
	# 	lock.release()

	time.sleep(0.01)

	outs.put((sample, np.random.rand(7)))


if __name__ == "__main__":
	x = []
	y = []

	a = np.zeros((8,))
	b = np.random.rand(8, 7, 6, 1)
	c = []

	print([a[i] * b[i] for i in range(8)])

	model = Sequential()

	model.add(Reshape((42,), input_shape=(7, 6, 1)))
	model.add(Dense(64, activation="relu"))
	model.add(Dense(64, activation="relu"))
	model.add(Dense(7, activation="relu"))

	model.compile(Adam(), loss="mse")

	print(model.predict(np.array([a[i] * b[i] for i in range(8)])))

	# test = np.array([np.random.rand(7, 6, 1) for _ in range(4)])
	# model.predict(test)

	# start = time.time()
	# with tf.device('/gpu:0'):
	# 	model.predict(test)
	# print(f"GPU : {time.time() - start}")

	# start = time.time()
	# with tf.device('/cpu:0'):
	# 	model.predict(test)
	# print(f"CPU : {time.time() - start}")

	setup_start = time.time()

	tasks = Queue()
	outs = Queue()

	lock = Lock()

	samples = [(np.random.rand(7), random.random()) for _ in range(BATCH_SIZE)]

	states, rewards = zip(*samples)

	print(np.argmax(states, axis=1), len(rewards))

	# for state in states:
	# 	tasks.put(state)

	# threads = [Thread(target=prepare_sample, args=[tasks, outs, lock]) for _ in range(BATCH_SIZE)]

	# for thread in threads:
	# 	thread.start()

	# for thread in threads:
	# 	thread.join()

	# for i in range(BATCH_SIZE):
	# 	state, q_values = outs.get()

	# 	x.append(state)
	# 	y.append(q_values)

	setup_end = time.time()

	print(f"Setup : {round(setup_end - setup_start, 3)}")

	print(len(x), " ", len(y), " ", counter)