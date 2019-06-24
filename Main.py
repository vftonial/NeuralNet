import copy
import random
import math
import statistics
import os
import sys
from decimal import Decimal

# seu valor de ativacao, a lista de pesos que entram nele
# casos especificos para o primeiro e ultimo layer


class Node:
	activation = 0
	weights = None
	error = 0
	grads = None
	batch_grads = list()
	m = list()
	v = list()

	def __init__(self, activation, weights=None, grads=None, batch_grads=list(), m=list(), v=list()):
		self.activation = activation
		self.weights = weights
		self.grads = grads
		self.batch_grads = batch_grads
		self.m = m
		self.v = v

	def my_str(self):
		result = ""
		for i in range(len(self.grads)):
			result += str("(" + "{0:.5f}".format(self.weights[i]) + ", " + "{0:.5f}".format(self.grads[i]) + ")" + ", ")
		return result[:-2]


class Instance:
	data = []
	result = []

	def __init__(self, data, result):
		self.data = data
		self.result = result

	def my_str(self):
		return ", ".join(map(str, self.data)) + "; " + ", ".join(map(str, self.result))


class NeuralNet:
	input_layer = []
	hidden_layers = []  # [[]]
	output_layer = []
	regularization = 0
	step = 1
	batch = False
	adam = False

	def my_str(self):
		lines = []
		for layer in self.hidden_layers:
			result = ""
			for node in layer[1:]:
				result += node.my_str() + "; "
			lines.append(result[:-2] + "\n")
		result = ""
		for node in self.output_layer:
			result += node.my_str() + "; "
		lines.append(result[:-2] + "\n")
		return "".join(map(str, lines))

	def create_input_layer(self, activation_list):
		self.input_layer = list()
		self.input_layer.append(Node(1))
		for x in activation_list:
			self.input_layer.append(Node(x))

	def create_hidden_layers(self, input_size, n_nodes, n_layers):
		self.hidden_layers = list()
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
		layer = list()
		layer.append(Node(1))
		for _ in range(n_nodes):
			weights = list(random.uniform(0, 1) for _ in range(n_weights + 1))
			grads = list(0 for _ in range(n_weights + 1))
			batch_grads = list([] for _ in range(n_weights + 1))
			m = list(0 for _ in range(n_weights + 1))
			v = list(0 for _ in range(n_weights + 1))
			layer.append(Node(0, weights, grads, batch_grads, m, v))
		return layer

	def output_layer_errors(self, expected_result):
		for index in range(len(self.output_layer)):
			f = self.output_layer[index].activation
			y = expected_result[index]
			self.output_layer[index].error = f - y

	def hidden_layer_errors(self, from_layer_i, to_layer_i):
		if from_layer_i > 0:
			from_layer = self.hidden_layers[from_layer_i]
			to_layer = self.hidden_layers[to_layer_i]
			self.layer_errors(from_layer, to_layer, 1)
			self.hidden_layer_errors(to_layer_i, to_layer_i - 1)

	def layer_errors(self, from_layer, to_layer, bias):
		for node_i in range(1, len(to_layer)):
			from_layer_size = len(from_layer)
			weights_x_error = list(
				from_layer[j].weights[node_i] * from_layer[j].error for j in range(bias, from_layer_size))
			node_activation = to_layer[node_i].activation
			to_layer[node_i].error = sum(weights_x_error) * node_activation * (1 - node_activation)

	def all_layers_errors(self, expected_result):
		self.output_layer_errors(expected_result)
		self.layer_errors(self.output_layer, self.hidden_layers[-1], 0)
		self.hidden_layer_errors(len(self.hidden_layers) - 1, len(self.hidden_layers) - 2)

	@staticmethod
	def gradient(from_layer, to_layer, first_node, second_node, lamb):
		if first_node == 0:
			lamb = 0
		return (from_layer[first_node].activation * to_layer[second_node].error) + (
				lamb * to_layer[second_node].weights[first_node])

	@staticmethod
	def m(m, g):
		return 0.9 * m + (1 - 0.9) * g

	@staticmethod
	def v(v, g):
		return 0.999 * v + (1 - 0.999) * g * g

	def mt(self, node, index, t):
		node.m[index] = self.m(node.m[index], node.grads[index])
		return node.m[index] / (1 - pow(0.9, t))

	def vt(self, node, index, t):
		node.v[index] = self.v(node.v[index], node.grads[index])
		return node.v[index] / (1 - pow(0.999, t))

	def adao(self, node, index, t):
		mt = self.mt(node, index, t)
		vt = self.vt(node, index, t)
		return mt / (math.sqrt(vt) + 0.000000001)

	def adjust_weights(self, alpha):
		self.adjust_weights_of_layer(alpha, self.hidden_layers[0], 1)
		for layer_i in range(1, len(self.hidden_layers) - 1):
			to_layer = self.hidden_layers[layer_i + 1]
			self.adjust_weights_of_layer(alpha, to_layer, 1)
		self.adjust_weights_of_layer(alpha, self.output_layer, 0)

	def adjust_weights_of_layer(self, alpha, to_layer, bias):
		for node_i in range(bias, len(to_layer)):
			node = to_layer[node_i]
			self.adjust_weights_of_node(alpha, node)

	def adjust_weights_of_node(self, alpha, node):
		for weight_i in range(len(node.weights)):
			if self.adam:
				node.weights[weight_i] = node.weights[weight_i] - alpha * self.adao(node, weight_i, self.step)
			else:
				node.weights[weight_i] = node.weights[weight_i] - alpha * node.grads[weight_i]

	def get_all_weights(self):
		weights = list()
		for layer in self.hidden_layers:
			for node in layer[1:]:
				for w in node.weights[1:]:
					weights.append(w)
		for node in self.output_layer:
			for w in node.weights[1:]:
				weights.append(w)
		return copy.deepcopy(weights)

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

	def calculate_gradients(self, lamb):
		self.calculate_gradients_of_layer(lamb, self.input_layer, self.hidden_layers[0], 1)
		for layer_i in range(0, len(self.hidden_layers) - 1):
			to_layer = self.hidden_layers[layer_i + 1]
			from_layer = self.hidden_layers[layer_i]
			self.calculate_gradients_of_layer(lamb, from_layer, to_layer, 1)
		self.calculate_gradients_of_layer(lamb, self.hidden_layers[-1], self.output_layer, 0)

	def calculate_gradients_of_layer(self, lamb, from_layer, to_layer, bias):
		for node_i in range(bias, len(to_layer)):
			node = to_layer[node_i]
			self.calculate_gradients_of_node(lamb, node, node_i, from_layer, to_layer)

	def calculate_gradients_of_node(self, lamb, node, node_i, from_layer, to_layer):
		for weight_i in range(len(node.weights)):
			grad = self.gradient(from_layer, to_layer, weight_i, node_i, lamb)
			if self.adam or self.batch:
				node.batch_grads[weight_i].append(grad)
				node.grads[weight_i] = sum(node.batch_grads[weight_i]) / len(node.batch_grads[weight_i])
			else:
				node.grads[weight_i] = grad

	@staticmethod
	def sigmoid(x):
		return 1 / (1 + math.exp(- x))


class Problem:
	instances = []
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
		file = open(filename, "r")
		lines = file.readlines()
		self.neural_net.regularization = float(lines[0])

		self.neural_net.create_input_layer([0 for _ in range(int(lines[1]) - 1)])

		hidden_layers = list(map(lambda x: int(x) - 1, lines[2:(len(lines) - 1)]))
		self.neural_net.create_hidden_layers(int(lines[1]) - 1, hidden_layers, len(hidden_layers))

		self.neural_net.create_output_layer(hidden_layers[-1], int(lines[-1]))

		file.close()

	def read_weights(self, filename):
		file = open(filename, "r")
		lines = file.readlines()
		layer = 0
		for line in lines[:-1]:
			nodes = line.split(";")
			node_index = 0
			for node in nodes:
				weights = list(map(float, node.split(",")))
				weight_index = 0
				for weight in weights:
					self.neural_net.hidden_layers[layer][node_index + 1].weights[weight_index] = float(weight)
					weight_index += 1
				node_index += 1
			layer += 1

		nodes = lines[-1].split(";")
		node_index = 0
		for node in nodes:
			weights = list(map(float, node.split(",")))
			weight_index = 0
			for weight in weights:
				self.neural_net.output_layer[node_index].weights[weight_index] = float(weight)
				weight_index += 1
			node_index += 1
		layer += 1

		file.close()

	def read_dataset(self, filename):
		file = open(filename, "r")
		lines = file.readlines()
		for line in lines:
			line = line.split(";")
			data = line[0].split(",")
			result = line[1].split(",")
			instance = Instance(list(map(float, data)), list(map(float, result)))
			self.instances.append(copy.deepcopy(instance))

	def backpropagation(self, n_layers, n_nodes, alpha, lamb, instances):
		size_input_layer = len(instances[0].data)

		self.neural_net = NeuralNet()
		# criar hidden layers
		self.neural_net.create_hidden_layers(size_input_layer, n_nodes, n_layers)
		# adicionar o ultimo layer com base no conjunto de entradas
		self.neural_net.create_output_layer(n_nodes[-1], self.output_size)

		self.backpropagation_with_no_init(alpha, lamb, instances, 1)

	def backpropagation_with_no_init(self, alpha, lamb, instances, iterations):
		for _ in range(iterations):
			inst_list = [instances[i::4] for i in range(4)]
			for inst in inst_list:
				for i in inst:
					self.neural_net.create_input_layer(i.data)
					self.propagate(i.result, lamb)
					if not self.neural_net.batch and not self.neural_net.adam:
						self.update(alpha)
						self.save_log()
				if self.neural_net.batch or self.neural_net.adam:
					self.update(alpha)
					self.save_log()
		self.neural_net.step = 1

	@staticmethod
	def save_results(alpha, n_layers, layers_size, mean, dev, lamb, filename):
		if not os.path.exists("results"):
			os.mkdir("results")
		file = open("results/" + filename + "Results.txt", "a", newline="\n")
		line = str(n_layers) + " " + str(layers_size) + " " + str(lamb) + " " + str(dev) + " " + str(mean) + " " + str(alpha) + "\n"
		file.write(line)
		file.close()

	def cross_validation(self, k, alpha, lamb, layers_n, layers_size):
		folds = Problem.create_folds(self.instances, k)
		scores = list()
		for fold in folds:
			self.backpropagation(layers_n, layers_size, alpha, lamb, fold["training"])
			scores.append(self.get_performance_of_net(fold["test"], lamb))

		result = dict()
		result["standardDeviation"] = statistics.pstdev(scores)
		result["meanPerformance"] = statistics.mean(scores)

		self.save_results(alpha, layers_n, layers_size, result["meanPerformance"], result["standardDeviation"], lamb, self.file_name)

		return result

	def get_performance_of_net(self, tests, lamb):
		result = list()
		for test in tests:
			self.neural_net.create_input_layer(copy.deepcopy(test.data))
			self.propagate(test.result, lamb)
			output = [node.activation for node in copy.deepcopy(self.neural_net.output_layer)]
			result.append([list(map(int, copy.deepcopy(test.result))), self.convert_output(output)])

		confusion_matrix = Confusion.make_confusion_matrix(copy.deepcopy(result), self)

		return Metrics.f1(confusion_matrix)

	@staticmethod
	def convert_output(output):
		result = [0 for _ in output]
		result[output.index(max(output))] = 1
		return result

	@staticmethod
	def create_folds(instances, k):
		size = len(instances)
		fold_size = int(size / k)
		rest_folds = size % k
		fold_sizes = [fold_size for _ in range(0, k)]
		for i in range(0, rest_folds):
			fold_sizes[i] += 1
		instances = set(copy.deepcopy(instances))

		folds = list()
		for fSize in fold_sizes:
			fold = set(random.sample(instances, fSize))
			instances -= fold
			folds.append(list(copy.deepcopy(fold)))

		result = list()
		for fold in folds:
			index = folds.index(fold)
			folds_list = copy.deepcopy(folds)
			del folds_list[index]
			fold_result = dict()
			fold_result["training"] = sum(folds_list, [])
			fold_result["test"] = copy.deepcopy(fold)
			result.append(fold_result)
		return result

	def propagate(self, expected_result, lamb):
		self.neural_net.propagate_input_layer()
		self.neural_net.propagate_hidden_layers()
		self.neural_net.propagate_output_layer()

		self.neural_net.all_layers_errors(expected_result)
		self.neural_net.calculate_gradients(lamb)

	def simple_propagate(self):
		self.neural_net.propagate_input_layer()
		self.neural_net.propagate_hidden_layers()
		self.neural_net.propagate_output_layer()

	def update(self, alpha):
		self.neural_net.adjust_weights(alpha)
		self.neural_net.step += 1

	def get_all_outputs(self):
		outputs = list()
		for i in range(self.output_size):
			output = [0 for _ in range(self.output_size)]
			output[i] = 1
			outputs.append(output)
		return outputs

	def numeric_validation(self, instances, lamb, epsilon):
		neuralnet_gradients = []
		derivative_cost = []
		derivative_errors = []

		for layer in self.neural_net.hidden_layers:
			for node in layer[1:]:
				for grad in node.grads:
					neuralnet_gradients.append(grad)
		for node in self.neural_net.output_layer:
			for grad in node.grads:
				neuralnet_gradients.append(grad)

		for layer in self.neural_net.hidden_layers:
			for node in layer[1:]:
				for weight in range(len(node.weights)):
					old_weight = node.weights[weight]
					node.weights[weight] = float(format(old_weight - epsilon, '.7g'))
					cost_neg = self.cost(instances, lamb, self.neural_net.get_all_weights())
					node.weights[weight] = float(format(old_weight + epsilon, '.7g'))
					cost_pos = self.cost(instances, lamb, self.neural_net.get_all_weights())
					node.weights[weight] = old_weight
					d = (cost_pos - cost_neg) / (2 * epsilon)
					derivative_cost.append(d)

		for node in self.neural_net.output_layer:
			for weight in range(len(node.weights)):
				old_weight = node.weights[weight]
				node.weights[weight] = float(format(old_weight - epsilon, '.7g'))
				cost_neg = self.cost(instances, lamb, self.neural_net.get_all_weights())
				node.weights[weight] = float(format(old_weight + epsilon, '.7g'))
				cost_pos = self.cost(instances, lamb, self.neural_net.get_all_weights())
				node.weights[weight] = old_weight
				d = (cost_pos - cost_neg) / (2 * epsilon)
				derivative_cost.append(d)

		for i in range(len(neuralnet_gradients)):
			derivative_errors.append(neuralnet_gradients[i] - derivative_cost[i])

		return derivative_errors

	def cost(self, instances, lamb, all_weights):
		summer = 0.0
		weights = list(map(lambda x: x * x, all_weights))
		for i in range(len(instances)):
			instance = instances[i]
			self.neural_net.create_input_layer(instance.data)
			self.simple_propagate()
			for k in range(len(instance.result)):
				y = instance.result[k]
				f = float(self.neural_net.output_layer[k].activation)
				ln_f = -10 if f == 0 else math.log(f)
				ln_1f = -10 if f == 1 else math.log(1.0 - f)
				summer += -y * ln_f - (1.0 - y) * ln_1f
		return (summer / len(instances)) + ((lamb * sum(weights)) / (2.0 * len(instances)))

	def save_log(self):
		if not os.path.exists("bruno_anderson"):
			os.mkdir("bruno_anderson")
		output = open("./bruno_anderson/log.txt", "a", newline="\n")
		output.write(str(self.neural_net.regularization) + '\n')
		output.writelines(str(self.neural_net.my_str()) + '\n\n')


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
			if int(line[0]) == 1:
				line = line[:len(line) - 1] + " 0"
			else:
				if int(line[0]) == 2:
					line = line[:len(line) - 1] + " 1"
				else:
					line = line[:len(line) - 1] + " 2"
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


class Confusion:
	@staticmethod
	def make_confusion_matrix(results, problem):
		possibilities = problem.get_all_outputs()
		mappingDict = dict()
		index = 0

		for possibility in possibilities:
			mappingDict["".join(map(str, possibility))] = index
			index += 1

		number_of_possibilities = len(possibilities)
		confusion_matrix = [[0 for _ in range(number_of_possibilities)] for _ in range(number_of_possibilities)]
		for index in range(0, len(results)):
			row = mappingDict["".join(map(str, results[index][0]))]
			column = mappingDict["".join(map(str, results[index][1]))]
			confusion_matrix[row][column] += 1

		return confusion_matrix


class Metrics:
	@staticmethod
	def recall(confusionMatrix):
		recalls = list()
		for i in range(0, len(confusionMatrix)):
			recalls.append(Metrics.recallOfAttribute(confusionMatrix, i))

		return statistics.mean(recalls)

	@staticmethod
	def precision(confusionMatrix):
		precisions = list()
		for i in range(0, len(confusionMatrix)):
			precisions.append(Metrics.precisionOfAttribute(confusionMatrix, i))

		return statistics.mean(precisions)

	@staticmethod
	def recallOfAttribute(confusionMatrix, index):
		tp = confusionMatrix[index][index]
		fn = sum(confusionMatrix[index]) - tp

		return tp / (tp + fn) if tp + fn != 0 else 0

	@staticmethod
	def precisionOfAttribute(confusionMatrix, index):
		tp = confusionMatrix[index][index]
		fp = sum([row[index] for row in confusionMatrix]) - tp

		return tp / (tp + fp) if tp + fp != 0 else 0

	@staticmethod
	def f1(confusionMatrix):
		precision = Metrics.precision(confusionMatrix)
		recall = Metrics.recall(confusionMatrix)
		score = 2 * (precision * recall) / (precision + recall) if precision + recall else 0
		return score


def pre_process():
	pima = "./data/pima.tsv"
	wine = "./data/wine.data"
	ionosphere = "./data/ionosphere.data"
	wdbc = "./data/wdbc.data"
	processor = PreProcess()
	processor.process_file(pima, PreProcess.format_pima)
	processor = PreProcess()
	processor.process_file(wine, PreProcess.format_wine)
	processor = PreProcess()
	processor.process_file(ionosphere, PreProcess.format_ionosphere)
	processor = PreProcess()
	processor.process_file(wdbc, PreProcess.format_wdbc)


def run(alpha, architectures, lambdas, file):
	problem = Problem()
	problem.read_normalized_file(file)
	for a in architectures:
		for l in lambdas:
			problem.cross_validation(5, 0.1, l, a[0], a[1])


def not_main():
	pre_process()

	architectures = list()
	lambdas = list()
	alpha = 0.3

	architectures.append([1, [1]])
	architectures.append([1, [2]])
	architectures.append([2, [1, 1]])
	architectures.append([2, [3, 3]])
	architectures.append([3, [2, 3, 2]])
	architectures.append([4, [3, 4, 4, 3]])

	lambdas.append(0.1)
	lambdas.append(0.25)

	# run(alpha, architectures, lambdas, "./normal_files/wdbcNormalizado.txt")
	# run(alpha, architectures, lambdas, "./normal_files/ionosphereNormalizado.txt")
	# run(alpha, architectures, lambdas, "./normal_files/wineNormalizado.txt")
	# run(alpha, architectures, lambdas, "./normal_files/pimaNormalizado.txt")


def run_with_given_network(network, weights, dataset):
	problem = Problem()
	problem.read_network(network)
	problem.read_weights(weights)
	problem.read_dataset(dataset)

	for instance in problem.instances:
		problem.neural_net.create_input_layer(instance.data)
		problem.propagate(instance.result, problem.neural_net.regularization)
		problem.update(0.01)
		if not os.path.exists("bruno_anderson"):
			os.mkdir("bruno_anderson")
		output = open("./bruno_anderson/results.txt", "a", newline="\n")
		output.write(instance.my_str() + '\n')
		output.write(str(problem.neural_net.regularization) + '\n')
		output.writelines(str(problem.neural_net.my_str()) + '\n\n')


def verify_gradients(network, weights, dataset, epsilon):
	problem = Problem()
	problem.read_network(network)
	problem.read_weights(weights)
	problem.read_dataset(dataset)

	for instance in problem.instances:
		problem.neural_net.create_input_layer(instance.data)
		problem.propagate(instance.result, problem.neural_net.regularization)
		if not os.path.exists("bruno_anderson"):
			os.mkdir("bruno_anderson")

	verification = problem.numeric_validation(problem.instances, problem.neural_net.regularization, float(epsilon))
	formatted_verification = convert_validation(verification, problem.neural_net)

	output = open("./bruno_anderson/verification.txt", "a", newline="\n")
	output.writelines(formatted_verification)


def convert_validation(validation, network):
	result = list()
	i = 0
	for layer in network.hidden_layers:
		aux = ""
		for node in layer[1:]:
			for _ in node.weights:
				aux += "{0:.5f}".format(validation[i]) + ", "
				i += 1
			aux = aux[:-2] + "; "
		aux = aux[:-2] + "\n"
		result.append(aux)

	aux = ""
	for node in network.output_layer:
		for _ in node.weights:
			aux += "{0:.5f}".format(validation[i]) + ", "
			i += 1
		aux = aux[:-2] + "; "
	aux = aux[:-2] + "\n"
	result.append(aux + "\n")

	return result


def backpropagation(network, weights, dataset, iter=1, typee=""):
	problem = Problem()
	problem.read_network(network)
	problem.read_weights(weights)
	problem.read_dataset(dataset)

	if typee == "adam":
		problem.neural_net.adam = True
		problem.neural_net.batch = True
	elif typee == "mini_batch":
		problem.neural_net.batch = True

	problem.backpropagation_with_no_init(0.1, problem.neural_net.regularization, problem.instances, int(iter))


def main(args):

	if len(args) == 3:
		run_with_given_network(args[0], args[1], args[2])
	elif len(args) == 4 and args[0] == "run":
		backpropagation(args[1], args[2], args[3])
	elif len(args) == 5 and args[0] == "numeric_validation":
		verify_gradients(args[1], args[2], args[3], args[4])
	elif len(args) == 5 and args[0] == "run":
		backpropagation(args[1], args[2], args[3], args[4])
	elif len(args) == 6 and args[0] == "run":
		backpropagation(args[1], args[2], args[3], args[4], args[5])


if __name__ == "__main__":
	main(sys.argv[1:])
