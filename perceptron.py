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

def activation(y, name: str = "relu"):
	if name == "relu":
		if y > 0:
			return y
		else:
			return 0

	elif name == "sigmoid":
		return 1/(1 + (np.e** (-y)))


if __name__ == "__main__":
	result = perceptron(1, 1, x3=-1, x4=-1)
	activated = activation(result, "relu")
	activated2 = activation(result, "sigmoid")

	print(f"y: {result}")
	print(activated)
	print(activated2)