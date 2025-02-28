# Perceptron code 
import numpy as np


def perceptron(*args: float, **kwargs: float):
	inputs1 = np.array(args)
	remaining_inps = []
	for i, j in kwargs.items():
		remaining_inps.append(j)

	inputs2 = np.array(remaining_inps)
	inputs = np.concatenate((inputs1, inputs2))

	print(inputs)


perceptron(1, 2, x3=3, x4=4)
 