import copy
import random


# seu valor de ativacao, a lista de pesos que entram nele
# casos especificos para o primeiro e ultimo layer
class Node:
    activation = 0
    weights = None

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
            self.propagate(instance)
            # comparar o valor de ativaçao do nodo de saida com o valor previsto na instancia e atualizar o seu peso
            self.atualization()

    def propagate(self, instance):
        # todo
