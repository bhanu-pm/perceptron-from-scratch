# Perceptron code 
import numpy as np


# Only takes in inputs
def perceptron(args: list):
	inputs = np.array(args)
	# print(inputs.shape)

	weights = np.ones((inputs.shape[0]+1,), dtype=float)
	# print(weights.shape)
	# print(weights)
	y = (weights[1:] @ inputs) + weights[0]
	return y

def activation(y, name: str = "relu"):
	if name == "relu":
		if y > 0:
			return y
		else:
			return 0

	elif name == "sigmoid":
		return 1/(1 + (np.e** (-y)))

# Only enter activation name and inputs
def predict(activation_name, *args, **kwargs):
	remaining_inps = []
	for i, j in kwargs.items():
		remaining_inps.append(j)

	inputs = np.concatenate((args, remaining_inps))

	y = perceptron(inputs)
	prediction = activation(y, activation_name)
	return prediction

def loss(y, y_hat):



if __name__ == "__main__":
	Y = predict("sigmoid", 1, 2, -1, -2)
