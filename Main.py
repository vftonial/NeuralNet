import copy
import random


# seu valor de ativacao, a lista de pesos que entram nele
# casos especificos para o primeiro e ultimo layer
class Node:
    activation = 0
    weights = []

    def __init__(self, activation, weights):
        self.activation = activation
        self.weights = weights


class Layer:
    nodes_vector = []


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

    def backpropagation(self, n_layers, n_nodes, instances):
        # inicializar os pesos da rede com não zero
        # para cada exemplo no treinamento
        #   propagar o exemplo na rede
        #   calcular o erro na camada de saida
        #   calcular os erros da camada oculta
        #   calcular os gradiantes
        #   ajustar os pesos
        # avaliar a performance no conjunto de treinamento, se ainda não ta decente roda dnv
        self.neural_net = NeuralNet()
        # adicionar o primeiro layer com base no conjunto de entradas
        for i in n_layers:
            self.neural_net.hidden_layers.append([])
            for _ in n_nodes:
                self.neural_net.hidden_layers[i].append(Node(0, list(random.uniform(0, 1) for _ in range(n_nodes))))

        # adicionaro o ultimo layer com base no conjunto de entradas
        self.neural_net.output_layer.append(Node(0, list(random.uniform(0, 1) for _ in range(n_nodes))) for _ in
                                            range(len(instances[0].result)))  # issae fe
