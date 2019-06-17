import copy
import random
import math


# seu valor de ativacao, a lista de pesos que entram nele
# casos especificos para o primeiro e ultimo layer
class Node:
    activation = 0
    weights = None
    error = 0

    def __init__(self, activation, weights):
        self.activation = activation
        self.weights = weights

    def __init__(self, activation):
        self.activation = activation


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
        first_layer = self.create_layer(input_size, n_nodes)
        self.hidden_layers.append(first_layer)
        # adicionar os hidden layers
        for _ in n_layers - 1:
            hidden_layer = self.create_layer(n_nodes, n_nodes)
            self.hidden_layers.append(hidden_layer)

    def create_output_layer(self, n_weights, n_nodes):
        self.output_layer = create_layer(n_weights, n_nodes)

    def create_layer(self, n_weights, n_nodes):
        layer = []
        for _ in range(n_nodes):
            weights = list(random.uniform(0, 1) for _ in range(n_weights))
            layer.append(Node(0, weights))
        return layer

    def output_layer_errors(self, expected_result):
        for index in range(len(self.output_layer)):
            f = output_layer[index].activation
            y = expected_result[index]
            output_layer[index].error = f - y

    def hidden_layer_errors(self, layer_i, next_layer):
        if(layer >= 0):
            current_layer = self.hidden_layers[layer_i]
            current_layer_size = len(current_layer)
            for node in range(1, current_layer_size): # i
                next_layer_nodes = len(next_layer) # j in N
                weights_x_error = list(next_layer[j].weights[node] * next_layer[j].error for j in range(1, next_layer_nodes))
                node_activation = current_layer[node].activation
                current_layer[node].error = sum(weights_x_error) * node_activation * (1 - node_activation)
            self.hidden_layer_errors(layer_i - 1, current_layer[1:])

    def all_layers_errors(self, expected_result):
        self.output_layer_errors(expected_result)
        self.hidden_layer_errors(len(self.hidden_layers) - 1, self.output_layer)

    def gradient(self, layer, first_node, second_node): # MAYBE ITS WRONG
        if(layer == len(self.hidden_layers)):
            current_layer = self.output_layer
        else:
            current_layer = self.hidden_layers[layer]
        previous_layer = self.hidden_layers[layer - 1]
        return previous_layer[first_node].activation * current_layer[second_node].error

    def ajust_weights(self, alpha):
        for layer_i in range(len(self.hidden_layers)):
            layer = self.hidden_layers[layer_i]
            for node_i in range(len(layer)):
                node = layer[node_i]
                for weight_i in range(len(node.weights)):
                    grad = self.gradient(layer_i, weight_i, node_i)
                    node.weights[weight_i] = node.weights[weight_i] - alpha * grad

    def cost(self, instances):
        summ = 0
        for i in range(len(instances)):
            instance = instances[i]
            for k in range(len(instance.result)):
                y = instance.result[k]
                f = self.output_layer[k].activation
                summ += -y * math.log(f) - (1 - y) * math.log(1 - f)  
        return summ / len(instances)

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
        self.propagate_layer(self.hidden_layer[-1], self.output_layer)

    def sigmoid(self, x):
        return 1 / (1 + math.exp(- x))


class Problem:
    instances = []
    neural_net = NeuralNet()

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

    def backpropagation(self, n_layers, n_nodes):
        # inicializar os pesos da rede com não zero
        # para cada exemplo no treinamento
        #   propagar o exemplo na rede
        #   calcular o erro na camada de saida
        #   calcular os erros da camada oculta
        #   calcular os gradiantes
        #   ajustar os pesos
        # avaliar a performance no conjunto de treinamento, se ainda não ta decente roda dnv

        size_input_layer = len(self.instances[0].data)
        size_output_layer = len(self.instances[0].result)

        self.neural_net = NeuralNet()
        # criar hidden layers
        self.neuralnet.create_hidden_layers(size_input_layer, n_nodes, n_layers)
        # adicionar o ultimo layer com base no conjunto de entradas
        self.neural_net.create_output_layer(n_nodes, size_output_layer)

        for instance in self.instances:
            self.neuralnet.create_input_layer(instance.data)
            self.propagate()
            # comparar o valor de ativaçao do nodo de saida com o valor previsto na instancia e atualizar o seu peso
            self.atualization()

    def propagate(self):
        self.neuralnet.propagate_input_layer()
        self.neuralnet.propagate_hidden_layers()
        self.neuralnet.propagate_output_layer()

    def atualization(self, expected_result):
        self.neuralnet.all_layers_errors(expected_result)
        self.neuralnet.ajust_weights(0.9, )
