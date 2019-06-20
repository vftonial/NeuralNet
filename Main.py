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

    def __init__(self, activation, weights=None):
        self.activation = activation
        self.weights = weights


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
        for x in activation_list:
            self.input_layer.append(Node(x))

    def create_hidden_layers(self, input_size, n_nodes, n_layers):
        # adicionar o primeiro layer com base no conjunto de entradas
        first_layer = self.create_layer(input_size, n_nodes[0])
        self.hidden_layers.append(first_layer)
        # adicionar os hidden layers
        for i in range(1, n_layers):
            hidden_layer = self.create_layer(n_nodes, n_nodes[i])
            self.hidden_layers.append(hidden_layer)

    def create_output_layer(self, n_weights, n_nodes):
        self.output_layer = self.create_layer(n_weights, n_nodes)

    def create_layer(self, n_weights, n_nodes):
        layer = []
        for _ in range(n_nodes):
            weights = list(random.uniform(0, 1) for _ in range(n_weights))
            layer.append(Node(0, weights))
        return layer

    def output_layer_errors(self, expected_result):
        for index in range(len(self.output_layer)):
            f = self.output_layer[index].activation
            y = expected_result[index]
            self.output_layer[index].error = f - y

    def hidden_layer_errors(self, layer_i, next_layer):
        if layer_i >= 0:
            current_layer = self.hidden_layers[layer_i]
            current_layer_size = len(current_layer)
            for node in range(1, current_layer_size):  # i
                next_layer_nodes = len(next_layer)  # j in N
                weights_x_error = list(
                    next_layer[j].weights[node] * next_layer[j].error for j in range(1, next_layer_nodes))
                node_activation = current_layer[node].activation
                current_layer[node].error = sum(weights_x_error) * node_activation * (1 - node_activation)
            self.hidden_layer_errors(layer_i - 1, current_layer[1:])

    def all_layers_errors(self, expected_result):
        self.output_layer_errors(expected_result)
        self.hidden_layer_errors(len(self.hidden_layers) - 1, self.output_layer)

    def gradient(self, layer, first_node, second_node, lamb):
        if first_node == 0:
            lamb = 0
        if layer == len(self.hidden_layers):
            current_layer = self.output_layer
        else:
            current_layer = self.hidden_layers[layer]
        previous_layer = self.hidden_layers[layer - 1]
        return (previous_layer[first_node].activation * current_layer[second_node].error) + (lamb * current_layer[second_node].weights[first_node])

    def adjust_weights(self, alpha, lamb):
        for layer_i in range(len(self.hidden_layers)):
            layer = self.hidden_layers[layer_i]
            for node_i in range(len(layer)):
                node = layer[node_i]
                for weight_i in range(len(node.weights)):
                    grad = self.gradient(layer_i, weight_i, node_i, lamb)
                    node.weights[weight_i] = node.weights[weight_i] - alpha * grad

    def cost(self, instances, lamb):
        summ = 0
        for i in range(len(instances)):
            instance = instances[i]
            for k in range(len(instance.result)):
                y = instance.result[k]
                f = self.output_layer[k].activation
                summ += -y * math.log(f) - (1 - y) * math.log(1 - f)
	
        return (summ / len(instances)) + ((lamb * sum(self.get_all_weights())) / (2 * len(instances)))

    def get_all_weights(self):
    	weights = list()
    	for layer in self.hidden_layers:
        	for node in layer[1:]:
        		for w in node.weights[1:]:
        			weigths.append(w)
        for node in self.output_layer:
    		for w in node.weights[1:]:
    			weigths.append(w)
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
        self.propagate_layer(self.hidden_layers[-1], self.output_layer)

    def sigmoid(self, x):
        return 1 / (1 + math.exp(- x))


class Problem:
    instances = []
    training = []
    test = []
    neural_net = NeuralNet()

    def read_normalized_file(self, filename):
        file = open(filename, "r")
        lines = file.readlines()
        for line in lines:
            line = line.split()
            data = line[:-1]
            result = line[-1]
            instance = Instance(list(map(float, data)), list(map(float, result)))
            self.instances.append(copy.deepcopy(instance))

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
        size_output_layer = len(self.instances[0].result)

        self.neural_net = NeuralNet()
        # criar hidden layers
        self.neural_net.create_hidden_layers(size_input_layer, n_nodes, n_layers)
        # adicionar o ultimo layer com base no conjunto de entradas
        self.neural_net.create_output_layer(n_nodes, size_output_layer)

        for _ in range(20):
	        for instance in self.instances:
	            self.neural_net.create_input_layer(instance.data)
	            self.propagate()
	            self.atualization(instance.result, alpha, lamb)
	        print(self.neural_net.cost(instances, lamb))

	def cross_validation(instances, attributes, k, forest_size):
		folds = Problem.create_folds(instances, k)
		scores = list()
		for i in range(0, k):
			problem = Problem()
			problem.attributes = copy.deepcopy(attributes)
			problem.instances = list(folds[i]["training"])
			problem.createForest(forestSize)
			scores.append(problem.getPerformanceOfForest(folds[i]["test"]))

	def create_folds(instances, k):
		size = len(instances)
		foldSize = int(size / k)
		restFolds = size % k
		foldSizes = [foldSize for i in range(0, k)]
		for i in range(0, restFolds):
			foldSizes[i] += 1
		instances = set(copy.deepcopy(instances))

		folds = list()
		for fSize in foldSizes:
			fold = set(random.sample(instances, fSize))
			instances -= fold
			folds.append(list(copy.deepcopy(fold)))

		result = list()
		for fold in folds:
			index = folds.index(fold)
			foldsList = copy.deepcopy(folds)
			del foldsList[index]
			foldResult = dict()
			foldResult["training"] = sum(foldsList, [])
			foldResult["test"] = copy.deepcopy(fold)
			result.append(foldResult)
		return result

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
        return path.split(".")[0].split("/")[-1]

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
        middle_file = "middle_files\\" + PreProcess.get_filename_from_path(filename) + "Intermediario.txt"
        outfile = open(middle_file, "w", newline="\n")
        for line in infile.readlines()[1:]:
            outfile.write(line)
        return middle_file

    @staticmethod
    def format_wine(filename):
        infile = open(filename, "r")
        if not os.path.exists("middle_files"):
            os.mkdir("middle_files")
        middle_file = "middle_files\\" + PreProcess.get_filename_from_path(filename) + "Intermediario.txt"
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
        middle_file = "middle_files\\" + PreProcess.get_filename_from_path(filename) + "Intermediario.txt"
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
        middle_file = "middle_files\\" + PreProcess.get_filename_from_path(filename) + "Intermediario.txt"
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
    # processor = PreProcess()
    # processor.process_file(wdbc, PreProcess.format_wdbc)
    # problem = Problem()
    # problem.read_normalized_file("normal_files\\wdbcNormalizado.txt")


if __name__ == "__main__":
    main()
