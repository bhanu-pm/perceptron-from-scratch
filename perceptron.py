# Perceptron code 
import numpy as np


class TrainingNeuron:
	def __init__(self, num_inputs, learning_rate:float = 0.01):
		self.weights = np.ones(num_inputs+1, dtype=float)
		self.learning_rate = learning_rate

	# Only takes in inputs
	def perceptron(self, inputs: list):
		inputs = np.array(inputs)

		y = (self.weights[1:] @ inputs) + self.weights[0]
		return y

	def activation(self, y, name: str = "relu"):
		if name == "relu":
			if y > 0:
				return y
			else:
				return 0

		elif name == "sigmoid":
			return 1/(1 + (np.e** (-y)))

	# Only enter activation name and inputs
	def forward(self, inputs, activation_name):
		logit = self.perceptron(inputs)
		prediction = self.activation(logit, activation_name)
		return prediction

	def loss(self, y, y_hat):
		return y - y_hat

	def update_rule(self, loss_val, inputs):
		self.weights[0] = self.weights[0] + (self.learning_rate * loss_val)
		self.weights[1:] = self.weights[1:] + (self.learning_rate * loss_val * inputs)


if __name__ == "__main__":
	num_inputs = int(input("Enter the number of inputs: integer: "))
	str_inputs = input("Enter the inputs, delimited by a comma: ").split(',')

	inputs = [int(i) for i in str_inputs]

	training = TrainingNeuron(num_inputs, 1e-3)
	y_hat = training.forward(inputs, "sigmoid")
	print(y_hat)
