import copy
import random
import math
import statistics
import os


# seu valor de ativacao, a lista de pesos que entram nele
# casos especificos para o primeiro e ultimo layer
class Node:
	activation = 0
	weights = None
	error = 0
	grads = None

	def __init__(self, activation, weights=None, grads=None):
		self.activation = activation
		self.weights = weights
		self.grads = grads


class Instance:
	data = []
	result = []

	def __init__(self, data, result):
		self.data = data
		self.result = result


class NeuralNet:
	input_layer = []
	hidden_layers = []  # [[]]
	output_layer = []
	regularization = 0

	def create_input_layer(self, activation_list):
		self.input_layer = list()
		self.input_layer.append(Node(1))
		for x in activation_list:
			self.input_layer.append(Node(x))

	def create_hidden_layers(self, input_size, n_nodes, n_layers):
		# adicionar o primeiro layer com base no conjunto de entradas
		first_layer = self.create_layer(input_size, n_nodes[0])
		self.hidden_layers.append(first_layer)
		# adicionar os hidden layers
		for i in range(1, n_layers):
			hidden_layer = self.create_layer(n_nodes[i - 1], n_nodes[i])
			self.hidden_layers.append(hidden_layer)

	def create_output_layer(self, n_weights, n_nodes):
		self.output_layer = self.create_layer(n_weights, n_nodes)[1:]

	@staticmethod
	def create_layer(n_weights, n_nodes):
		layer = []
		layer.append(Node(1))
		for _ in range(n_nodes):
			weights = list(random.uniform(0, 1) for _ in range(n_weights + 1))
			grads = list(0 for _ in range(n_weights + 1))
			layer.append(Node(0, weights, grads))
		return layer

	def output_layer_errors(self, expected_result):
		for index in range(len(self.output_layer)):
			f = self.output_layer[index].activation
			y = expected_result[index]
			self.output_layer[index].error = f - y

	def hidden_layer_errors(self, from_layer_i, to_layer_i):
		if(from_layer_i > 0):
			from_layer = self.hidden_layers[from_layer_i]
			to_layer = self.hidden_layers[to_layer_i]
			self.layer_errors(from_layer, to_layer)
			self.hidden_layer_errors(to_layer_i, to_layer_i - 1)

	def layer_errors(self, from_layer, to_layer):
		for node_i in range(1, len(to_layer)):
			from_layer_size = len(from_layer)
			weights_x_error = list(
				from_layer[j].weights[node_i] * from_layer[j].error for j in range(1, from_layer_size))
			node_activation = to_layer[node_i].activation
			to_layer[node_i].error = sum(weights_x_error) * node_activation * (1 - node_activation)

	def all_layers_errors(self, expected_result):
		self.output_layer_errors(expected_result)
		self.layer_errors(self.output_layer, self.hidden_layers[-1])
		self.hidden_layer_errors(len(self.hidden_layers) - 1, len(self.hidden_layers) - 2)

	def gradient(self, from_layer, to_layer, first_node, second_node, lamb):
		if first_node == 0:
			lamb = 0
		return (from_layer[first_node].activation * to_layer[second_node].error) + (lamb * to_layer[second_node].weights[first_node])

	def adjust_weights(self, alpha, lamb):
		self.adjust_weights_of_layer(alpha, lamb, self.input_layer, self.hidden_layers[0])
		for layer_i in range(1, len(self.hidden_layers) - 1):
			to_layer = self.hidden_layers[layer_i + 1]
			from_layer = self.hidden_layers[layer_i]
			self.adjust_weights_of_layer(alpha, lamb, from_layer, to_layer)
		self.adjust_weights_of_layer(alpha, lamb, self.hidden_layers[-1], self.output_layer)

	def adjust_weights_of_layer(self, alpha, lamb, from_layer, to_layer):
		for node_i in range(1, len(to_layer)):
			node = to_layer[node_i]
			self.adjust_weights_of_node(alpha, lamb, node, node_i, from_layer, to_layer)

	def adjust_weights_of_node(self, alpha, lamb, node, node_i, from_layer, to_layer):
		for weight_i in range(len(node.weights)):
			grad = self.gradient(from_layer, to_layer, weight_i, node_i, lamb)
			node.grads[weight_i] = grad
			node.weights[weight_i] = node.weights[weight_i] - alpha * grad

	def cost(self, instances, lamb, all_weights):
		summer = 0.0
		for i in range(len(instances)):
			instance = instances[i]
			for k in range(len(instance.result)):
				y = instance.result[k]
				f = float(self.output_layer[k].activation)
				ln_f = -10 if f == 0 else math.log(f)
				ln_1f = -10 if f == 1 else math.log(1.0 - f)
				summer += -y * ln_f - (1.0 - y) * ln_1f
		return (summer / len(instances)) + ((lamb * sum(all_weights)) / (2.0 * len(instances)))

	def get_all_weights(self):
		weights = list()
		for layer in self.hidden_layers:
			for node in layer[1:]:
				for w in node.weights[1:]:
					weights.append(w)
		for node in self.output_layer:
			for w in node.weights[1:]:
				weights.append(w)
		return weights

	def propagate_layer(self, from_layer, to_layer):
		for to_node in to_layer[1:]:
			to_node.activation = 0
			for from_node_index in range(len(from_layer)):
				from_node = from_layer[from_node_index]
				to_node.activation += from_node.activation * to_node.weights[from_node_index]
			to_node.activation = self.sigmoid(to_node.activation)

	def propagate_input_layer(self):
		self.propagate_layer(self.input_layer, self.hidden_layers[0])

	def propagate_hidden_layers(self):
		for layer_index in range(len(self.hidden_layers) - 1):
			from_layer = self.hidden_layers[layer_index]
			to_layer = self.hidden_layers[layer_index + 1]
			self.propagate_layer(from_layer, to_layer)

	def propagate_output_layer(self):
		to_layer = self.output_layer
		from_layer = self.hidden_layers[-1]
		for to_node in to_layer:
			to_node.activation = 0
			for from_node_index in range(len(from_layer)):
				from_node = from_layer[from_node_index]
				to_node.activation += from_node.activation * to_node.weights[from_node_index]
			to_node.activation = self.sigmoid(to_node.activation)

	@staticmethod
	def sigmoid(x):
		return 1 / (1 + math.exp(- x))

	def numeric_validation(self, instances, lamb, epsilon):
		neuralnet_gradients = []
		derivative_cost = []
		derivative_errors = []

		for layer in self.hidden_layers:
			for node in layer[1:]:
				for grad in node.grads[1:]:
					neuralnet_gradients.append(grad)
		for node in self.output_layer:
			for grad in node.grads[1:]:
				neuralnet_gradients.append(grad)

		weights_eps_pos = self.get_all_weights()
		for i in range(len(weights_eps_pos)):
			weights_eps_neg = weights_eps_pos
			weights_eps_neg[i] = weights_eps_pos[i] - epsilon
			weights_eps_pos[i] = weights_eps_pos[i] + epsilon
			d = (self.cost(instances, lamb, weights_eps_pos) - self.cost(instances, lamb, weights_eps_neg)) / 2*epsilon
			derivative_cost.append(d)
			derivative_errors.append(neuralnet_gradients[i] - derivative_cost[i])
			weights_eps_pos = self.get_all_weights()
		return derivative_errors


class Problem:
	instances = []
	training = []
	test = []
	output_size = 0
	neural_net = NeuralNet()
	file_name = ""

	def read_normalized_file(self, filename):
		file = open(filename, "r")
		self.file_name = filename.split("/")[2].split("Normalizado")[0]
		lines = file.readlines()
		data = []
		result = []
		for line in lines:
			line = line.split()
			data.append(line[:-1])
			result.append(int(line[-1]))
		self.output_size = max(result) + 1
		i = 0
		for d in data:
			result_list = [0 for _ in range(self.output_size)]
			result_list[result[i]] = 1
			instance = Instance(list(map(float, d)), list(map(float, result_list)))
			self.instances.append(copy.deepcopy(instance))
			i += 1

	def read_network(self, filename):
		layer = 0
		file = open(filename, "r")
		lines = file.readlines()
		self.neural_net.regularization = int(lines[0])

		for i in range(int(lines[1]) + 1):  # +1 para adicionar o termo de bias
			self.neural_net.input_layer.append(Node(0, []))

		for line in lines[2:len(lines) - 1]:  # aparentemente corta antes da ultima entrada
			for i in range(int(line) + 1):  # +1 para adicionar o termo de bias
				self.neural_net.hidden_layers.append([])
				self.neural_net.hidden_layers[layer].append(
					Node(0, []))  # certamente esta errado, mas no meu teste funciona
			layer = layer + 1

		for i in range(int(lines[len(lines)])):
			self.neural_net.output_layer.append(Node(0, []))

	def read_weights(self, filename):
		file = open(filename, "r")
		lines = file.readlines()
		layer = 0
		node_index = 0
		for line in lines[1:]:
			if ";" in line:
				line = line.split(";")
				for node in line:
					for weight in node:
						self.neural_net.hidden_layers[layer][node_index].weights.append(int(weight))
						node_index = node_index + 1
					node_index = 0

			else:
				self.neural_net.hidden_layers[layer][0].weights.append(list(map(float, line)))  # desculpafe
			layer = layer + 1

	def read_dataset(self, filename):
		file = open(filename, "r")
		lines = file.readlines()
		for line in lines:
			line = line.split(";")
			data = line[0].split(" ")
			data.remove(data[-1])
			result = line[1].strip("\n").split(" ")
			result.remove(result[0])
			instance = Instance(list(map(float, data)), list(map(float, result)))
			self.instances.append(copy.deepcopy(instance))

	def backpropagation(self, n_layers, n_nodes, alpha, lamb):
		size_input_layer = len(self.instances[0].data)

		self.neural_net = NeuralNet()
		# criar hidden layers
		self.neural_net.create_hidden_layers(size_input_layer, n_nodes, n_layers)
		# adicionar o ultimo layer com base no conjunto de entradas
		self.neural_net.create_output_layer(n_nodes[-1], self.output_size)

		for _ in range(100):
			for instance in self.instances:
				self.neural_net.create_input_layer(instance.data)
				self.propagate()
				self.atualization(instance.result, alpha, lamb)
			j = self.neural_net.cost(self.instances, lamb, self.neural_net.get_all_weights())
			self.save_results(j, alpha, lamb, self.file_name)
		print(self.neural_net.numeric_validation(self.instances, lamb, 0.00000005))

	@staticmethod
	def save_results(j, alpha, lamb, filename):
		if not os.path.exists("results"):
			os.mkdir("results")
		file = open("results/" + filename + "Results.txt", "a", newline="\n")
		line = str(j) + " " + str(alpha) + " " + str(lamb) + "\n"
		file.write(line)
		file.close()

	# def cross_validation(instances, attributes, k, forest_size):
	#     folds = Problem.create_folds(instances, k)
	#     scores = list()
	#     for i in range(0, k):
	#         problem = Problem()
	#         problem.attributes = copy.deepcopy(attributes)
	#         problem.instances = list(folds[i]["training"])
	#         problem.createForest(forestSize)
	#         scores.append(problem.getPerformanceOfForest(folds[i]["test"]))
	#
	# def create_folds(instances, k):
	#     size = len(instances)
	#     fold_size = int(size / k)
	#     rest_folds = size % k
	#     fold_sizes = [fold_size for i in range(0, k)]
	#     for i in range(0, rest_folds):
	#         fold_sizes[i] += 1
	#     instances = set(copy.deepcopy(instances))
	#
	#     folds = list()
	#     for fSize in fold_sizes:
	#         fold = set(random.sample(instances, fSize))
	#         instances -= fold
	#         folds.append(list(copy.deepcopy(fold)))
	#
	#     result = list()
	#     for fold in folds:
	#         index = folds.index(fold)
	#         folds_list = copy.deepcopy(folds)
	#         del folds_list[index]
	#         fold_result = dict()
	#         fold_result["training"] = sum(folds_list, [])
	#         fold_result["test"] = copy.deepcopy(fold)
	#         result.append(fold_result)
	#     return result

	def propagate(self):
		self.neural_net.propagate_input_layer()
		self.neural_net.propagate_hidden_layers()
		self.neural_net.propagate_output_layer()

	def atualization(self, expected_result, alpha, lamb):
		self.neural_net.all_layers_errors(expected_result)
		self.neural_net.adjust_weights(alpha, lamb)


class PreProcess:
	data = None

	def calculate_max_min_mean_deviation(self, middle_file):
		infile = open(middle_file, "r")
		lines = infile.readlines()
		# total of columns, att + class
		metadata = len(lines[0].split())
		#              min      max   mean  dev
		self.data = [[9999.0, -9999.0, 0.0, 0.0] for _ in range(int(metadata))]
		raw_data = [[] for _ in range(int(metadata))]
		for line in lines[1:]:
			i = 0
			for number in line.split():
				raw_data[i].append(float(number))
				if self.data[i][0] > float(number):
					self.data[i][0] = float(number)
				if self.data[i][1] < float(number):
					self.data[i][1] = float(number)
				self.data[i][2] += float(number)
				i += 1
		for i in range(len(self.data)):
			self.data[i][2] = float("{0:.3f}".format(self.data[i][2] / float(len(raw_data[0]))))
		infile.close()
		i = 0
		for att in raw_data:
			self.data[i][3] = float(statistics.stdev(att))
			i += 1

	def norma_and_stand(self, middle_file, real_name):
		infile = open(middle_file, "r")
		if not os.path.exists("normal_files"):
			os.mkdir("normal_files")
			os.mkdir("standard_files")
		normal_file = open("normal_files\\" + real_name + "Normalizado.txt", "w", newline="\n")
		standard_file = open("standard_files\\" + real_name + "Padronizado.txt", "w", newline="\n")
		for line in infile.readlines():
			i = 0
			for number in line.split():
				if i < len(line.split()) - 1:
					normal_number = (float(number) - self.data[i][0]) / (self.data[i][1] - self.data[i][0])
					standard_number = (float(number) - self.data[i][2]) / self.data[i][3]
					normal_file.write("{0:.6f}".format(normal_number) + " ")
					standard_file.write("{0:.6f}".format(standard_number) + " ")
					i += 1
				else:
					normal_file.write(str(number))
					standard_file.write(str(number))
			normal_file.write("\n")
			standard_file.write("\n")
		normal_file.close()
		standard_file.close()
		infile.close()

	@staticmethod
	def get_filename_from_path(path):
		return path.split(".")[1].split("/")[-1]

	def process_file(self, filename, process_function):
		real_name = PreProcess.get_filename_from_path(filename)
		middle_file = process_function(filename)
		self.calculate_max_min_mean_deviation(middle_file)
		self.norma_and_stand(middle_file, real_name)

	@staticmethod
	def format_pima(filename):
		infile = open(filename, "r")
		if not os.path.exists("middle_files"):
			os.mkdir("middle_files")
		middle_file = "middle_files/" + PreProcess.get_filename_from_path(filename) + "Intermediario.txt"
		outfile = open(middle_file, "w", newline="\n")
		for line in infile.readlines()[1:]:
			outfile.write(line)
		return middle_file

	@staticmethod
	def format_wine(filename):
		infile = open(filename, "r")
		if not os.path.exists("middle_files"):
			os.mkdir("middle_files")
		middle_file = "middle_files/" + PreProcess.get_filename_from_path(filename) + "Intermediario.txt"
		outfile = open(middle_file, "w", newline="\n")
		lines = infile.readlines()
		for line in lines:
			line = line.replace(",", " ")
			line = line[:len(line) - 1] + " " + line[0]
			line = line[1:]
			outfile.write(line + "\n")
		return middle_file

	@staticmethod
	def format_ionosphere(filename):
		infile = open(filename, "r")
		if not os.path.exists("middle_files"):
			os.mkdir("middle_files")
		middle_file = "middle_files/" + PreProcess.get_filename_from_path(filename) + "Intermediario.txt"
		outfile = open(middle_file, "w", newline="\n")
		lines = infile.readlines()
		for line in lines:
			line = line[:1] + line[3:]
			line = line.replace(",", " ")
			line = line[:-1]
			if line[-1] == "g":
				line = line[:-1]
				line = line + "1"
			else:
				line = line[:-1]
				line = line + "0"
			outfile.write(line + "\n")
		return middle_file

	@staticmethod
	def format_wdbc(filename):
		infile = open(filename, "r")
		if not os.path.exists("middle_files"):
			os.mkdir("middle_files")
		middle_file = "middle_files/" + PreProcess.get_filename_from_path(filename) + "Intermediario.txt"
		outfile = open(middle_file, "w", newline="\n")
		lines = infile.readlines()
		for line in lines:
			line = line.replace(",", " ")
			line = line[line.find(" ") + 1:len(line) - 1]
			line = line[1:] + " " + line[0]
			if line[-1] == "B":
				line = line[:-1]
				line = line + "0"
			else:
				line = line[:-1]
				line = line + "1"
			outfile.write(line + "\n")
		return middle_file


def main():
	pima = "./data/pima.tsv"
	wine = "./data/wine.data"
	ionosphere = "./data/ionosphere.data"
	wdbc = "./data/wdbc.data"
	# processor = PreProcess()
	# processor.process_file(pima, PreProcess.format_pima)
	# processor = PreProcess()
	# processor.process_file(wine, PreProcess.format_wine)
	# processor = PreProcess()
	# processor.process_file(ionosphere, PreProcess.format_ionosphere)
	processor = PreProcess()
	processor.process_file(wdbc, PreProcess.format_wdbc)
	problem = Problem()
	problem.read_normalized_file("./normal_files/wdbcNormalizado.txt")
	problem.backpropagation(4, [3, 2, 3, 2], 0.3, 0.0003)


if __name__ == "__main__":
	main()
