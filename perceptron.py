# Perceptron code 
import pandas as pd
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

	def backward(self, inputs, y, y_hat):
		loss_val = self.loss(y, y_hat)
		self.update_rule(loss_val, inputs)
		return loss_val


def train_loop(train_dataset, num_inputs, activation_name, epochs):
	avg_loss = 0.0
	
	training = TrainingNeuron(num_inputs, 1e-5)
	for epoch in range(epochs):
		counter = 0
		for i in train_ds:
			counter += 1
			# split into X and Y
			x, y = i[:-1], i[-1]

			# forward pass
			y_hat = training.forward(x, activation_name)

			# calculate loss
			# backward pass to update weights
			loss_val = training.backward(x, y, y_hat)
			avg_loss += loss_val

			if counter % 10 == 0:
				print(f"Average loss per 10 training examples: {avg_loss/10}")
				avg_loss = 0.0

		# print(f"Average loss per epoch {avg_loss_epoch/100}")

	print(f"These are the weights: {training.weights}")

if __name__ == "__main__":
	### TRAINING LOOP
	ds = pd.read_csv("ds.csv")
	# print(ds.head(10))
	dataset = ds.to_numpy()

	# Train test split
	train_ds = dataset[:80, :]
	test_ds = dataset[80:, :]

	num_inputs = 3
	activation_name = "relu"
	epochs = 200

	train_loop(train_ds, num_inputs, activation_name, epochs)

