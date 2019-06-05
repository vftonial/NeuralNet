# seu valor de ativacao, a lista de pesos que saem dele e a lista de nodos aos quais ele se liga
# casos especificos para o primeiro e ultimo layer
class Node:
    activation = 0
    weights = []
    nodes = []


class Layer:
    nodes_vector = []


class Instances:
    data = []
    result = []


class NeuralNet:
    input_layer = []
    hidden_layers = [[]]
    output_layer = []
    regularization = 0

    def backpropagation(self, filename):
        # inicializar os pesos da rede com n zero
        # para cada exemplo no treinamento
        #   propagar o exemplo na rede
        #   calcular o erro na camada de saida
        #   calcular os erros da camada oculta
        #   calcular os gradiantes
        #   ajustar os pesos
        # avaliar a performance no conjunto de treinamento, se ainda não ta decente roda dnv
        self.read_network(filename)

    def read_network(self, filename):
        layer = 0
        file = open(filename, "r")
        lines = file.readlines()
        regularization = int(lines[0])

        for i in range(int(lines[1])):
            self.input_layer.append(Node())

        for line in lines[2:len(lines) - 1]:  # aparentemente corta antes da ultima entrada
            for i in range(int(line)):
                self.hidden_layers.append([])
                self.hidden_layers[layer][i].append(Node())  # certamente esta errado
            layer = layer + 1

        for i in range(int(lines[len(lines)])):
            self.output_layer.append(Node())

    def read_weights(self, filename):
        file = open(filename, "r")
        lines = file.readlines()
        index = 0
        for line in lines:
            if index == 0:



    def read_dataset(self, filename):
        file = open(filename, "r")
