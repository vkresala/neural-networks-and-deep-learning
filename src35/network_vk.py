import numpy as np
from math import exp

class Network:
    def __init__(self, sizes):
        """seizes = vstupni neurony + vrstvy site"""
        self.sizes = sizes
        # weights = pro kazdou aktualni bumku se jedna o soubor vah pro kazdy
        # neuron z aktualni vrstvy
        self.weights = [np.random.randn(y,z) for y, z in zip(sizes[1:], sizes[:-1])]
        # bias = jedno cislo pro kazdy aktualni neuron
        self.biases = [np.random.randn(x, 1) for x in sizes[1:]]


    def sigmoid_f(w, x, b):
        # 1 / 1+exp(−∑jwjxj−b)
        print("sig1")
        print("w = ", w)
        print("x = ", x)
        print("b = ", b)
        return 1.0 / (1.0 + exp( -np.dot(w, x) - b))

    # def feed_forward(self, x):
    #     for w, b in zip(self.weights, self.biases):
    #         x = self.sigmoid_f(w, x, b)
    #     return x

    def feed_forward(self, xs):
        for ws, bs in zip(self.weights, self.biases):
            new_xs = []
            for w, b in zip(ws, bs):
                new_xs.append(Network.sigmoid_f(w, xs, b))
            xs = np.array(new_xs)
        return xs


test = Network([4,3,2])

print("weights")
for w in test.weights:
    print(w.shape)

print("biases")
for b in test.biases:
    print(b.shape)

test_input = np.random.uniform(low=0.0, high=1.0, size=(4,1))

x = test.feed_forward(test_input)
print(x)