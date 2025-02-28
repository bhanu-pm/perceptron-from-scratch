# Perceptron code 
import numpy as np


def perceptron(*args: float, **kwargs: float):
	inputs1 = np.array(args)
	remaining_inps = []
	for i, j in kwargs.items():
		remaining_inps.append(j)

	inputs2 = np.array(remaining_inps)
	inputs = np.concatenate((inputs1, inputs2))
	# print(inputs.shape)

	weights = np.ones(inputs.shape, dtype=float)
	# print(weights.shape)
	# print(weights)
	weights_0 = 0
	y = (weights @ inputs) + weights_0

	return y

perceptron(1, 2, x3=3, x4=4)


