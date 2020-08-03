import numpy as np
import random
import json


class NeuralNetwork:
    """
    Neural Network Toy for python, by Arthur Vergaças.
    Based on Daniel Shiffman's library
    """

    def __init__(self, inputs, outputs, *hidden):
        """
        Constructor function

        Args:
            inputs (int): number of inputs neurons
            outputs (int): number of outputs
            *args (int): the number of nodes of all hidden layers
        """
        self.inputs_num = inputs

        self.hidden_num = []
        for arg in hidden:
            self.hidden_num.append(arg)
        self.hidden_layers = len(self.hidden_num)

        self.outputs_num = outputs

        # create an array of all hidden layer weights
        self.weights = []
        for i in range(self.hidden_layers):
            if i == 0:
                self.weights.append(np.random.rand(
                    self.hidden_num[0], self.inputs_num))
                # self.weights.append(
                #     np.full((self.hidden_num[0], self.inputs_num), 5.0))
            else:
                self.weights.append(np.random.rand(
                    self.hidden_num[i], self.hidden_num[i - 1]))

        self.weights_ho = np.random.rand(
            self.outputs_num, self.hidden_num[self.hidden_layers - 1])
        # self.weights_ho = np.full((
        #     self.outputs_num, self.hidden_num[self.hidden_layers - 1]), 1.0)

        self.bias_h = []
        for i in range(self.hidden_layers):
            self.bias_h.append(np.random.rand(self.hidden_num[i], 1))
            # self.bias_h.append(np.full((self.hidden_num[i], 1), 3.0))

        self.bias_o = np.random.rand(self.outputs_num, 1)
        # self.bias_o = np.full((self.outputs_num, 1), 2.0)

        self.learning_rate = 0.1

        # the derivative of the sigmoid function vectorized
        self.dsig_vec = np.vectorize(self.dsigmoid)

    def sigmoid(self, m):
        """
        Sigmoid function as an activation function.
        Since it uses numpy bult-in functions, it is faster and apllied to all elements of the array.


        Args:
            x (NumPy array): the matrix that numpy will aplly the function

        Returns:
            NumPy array: the matrix with the function applied to it
        """
        return 1 / (1 + np.exp(-m))

    def dsigmoid(self, m):
        """
        The derivative of the sigmoid function.
        Used to determine how much the weights need to change.
        To actually aplly it to all elements of the matrix, it is necessary to vectorize the function.
        To do so, use np.vectorize(dsigmoid)(matrix)

        Args:
            m (NumPy Array): the matrix that the function will be apllied

        Returns:
            NumPy array: actually, to return a numpy array its necessary to
            first vectorize it using np.vectorize(dsigmoid)(matrix)
        """
        return m * (1 - m)

    def predict(self, input_array):
        """
        Predicts the output of a certain input

        Args:
            input_array (array): An array of inputs suited to the Neural Network training
        Returns:
            returns a Python list with the guesses
        """

        # FEEDFORWARD ALGORITHM
        # the algorithm is the same used in the train function
        # just stored it here to have a way to predict values

        # create the numpy arrays
        inputs = np.array(input_array, ndmin=2).T

        # FEEDFORWARD ALGORITHM

        # define the hidden layer outputs
        # define first hidden layer output
        hidden_output = self.weights[0] @ inputs
        hidden_output = hidden_output + self.bias_h[0]
        # apply activation function
        hidden_output = self.sigmoid(hidden_output)

        # loops for the layers that interact only with other hidden layers
        for i in range(1, self.hidden_layers):
            hidden_output = self.weights[i] @ hidden_output
            hidden_output = hidden_output + self.bias_h[i]
            hidden_output = self.sigmoid(hidden_output)

        # define the output layer outputs
        # hidden layer -> output layer
        output_output = self.weights_ho @ hidden_output
        output_output = output_output + self.bias_o
        # apply activation function
        output_output = self.sigmoid(output_output)

        output_output.tolist()

        return output_output

    def train(self, input_array, label_array):
        """
        Trains the neural network with supervised learning
        The 'train' method trains the neural network just once, so it's necessary
        to train it within a loop
        It's important to recall that the given array of inputs must have a shape of (num_of_inputs, 1).
        In other words, just one column, and num_of_inputs rows.
        The numpy.array() method, when provided a python list, yelds an single line array.
        To avoid it, give to the method transposed ndarrays

        Args:
            input_array (array): A trasnposed ndarray (numpy array) of inputs
            label_array (array): A transposed ndarray (numpy array) of labels, according to the inputs
        """

        # create the numpy arrays
        inputs = input_array
        labels = label_array

        # FEEDFORWARD ALGORITHM
        # even i already have a function that does this process, i wanted to have
        # it here also so i can access the variables for the backpropragation algorithm

        # array to keep track of all hidden outputs
        hidden_opts = []

        # define the hidden layer outputs
        # define first hidden layer output
        hidden_output = self.weights[0] @ inputs
        hidden_output = hidden_output + self.bias_h[0]
        # apply activation function
        hidden_output = self.sigmoid(hidden_output)
        hidden_opts.append(hidden_output)

        # loops for the layers that interact only with other hidden layers
        for i in range(1, self.hidden_layers):
            hidden_output = self.weights[i] @ hidden_output
            hidden_output = hidden_output + self.bias_h[i]
            hidden_output = self.sigmoid(hidden_output)
            hidden_opts.append(hidden_output)

        # revert the list so the last output comes first
        hidden_opts.reverse()

        # define the output layer outputs
        # hidden layer -> output layer
        output_output = self.weights_ho @ hidden_output
        output_output = output_output + self.bias_o
        # apply activation function
        output_output = self.sigmoid(output_output)

        # BACKPROPAGATION ALGORITHM

        # calculate how much the network has missed
        # ERROR = TARGET - OUTPUT

        # output layer error
        output_error = labels - output_output

        # calculate the gradient of the formula to backpropagate the error to the first layers
        # hardcore math and calculus, might see later
        # each layer has its own gradient, as well its own formula
        # using vectorize for reasons described in the dsigmoid function
        output_gradient = self.dsig_vec(output_output)
        output_gradient *= output_error
        output_gradient *= self.learning_rate

        # calculate how much the weights must change
        # using transposed matrix because it's going backwards
        weights_ho_delta = output_gradient @ hidden_output.T

        # adjust weights
        self.weights_ho += weights_ho_delta
        # adjust bias (it's just the gradient)
        self.bias_o += output_gradient

        # hidden layer error
        hidden_error = self.weights_ho.T @ output_error

        # calculate hidden layer gradient (same proccess)
        hidden_gradient = self.dsig_vec(hidden_opts[0])
        hidden_gradient *= hidden_error
        hidden_gradient *= self.learning_rate

        # hidden deltas
        if self.hidden_layers == 1:
            weights_h_delta = hidden_gradient @ inputs.T
        else:
            weights_h_delta = hidden_gradient @ hidden_opts[1].T

        # adjust weights
        self.weights[self.hidden_layers - 1] += weights_h_delta
        # adjust bias
        self.bias_h[self.hidden_layers - 1] += hidden_gradient

        # loops for the layers that interact only with other hidden layers
        for i in range(1, self.hidden_layers):
            hidden_error = self.weights[self.hidden_layers -
                                        i].T @ hidden_error

            hidden_gradient = self.dsig_vec(hidden_opts[i])
            hidden_gradient *= hidden_error
            hidden_gradient *= self.learning_rate

            if i == self.hidden_layers - 1:
                weights_h_delta = hidden_gradient @ inputs.T
            else:
                weights_h_delta = hidden_gradient @ hidden_opts[i + 1].T

            self.weights[self.hidden_layers - (i + 1)] += weights_h_delta

            self.bias_h[self.hidden_layers - (i + 1)] += hidden_gradient

    def save(self, file_path):
        """
        Serializes the weights and biases of the class.
        Use it to save training

        Args:
            file_path (str): path to the file that will contain the serialized data
        """

        temp_w = []
        for w in self.weights:
            temp_w.append(w.tolist())
        temp_b = []
        for b in self.bias_h:
            temp_b.append(b.tolist())

        data = {
            "weights": temp_w,
            "weights_ho": self.weights_ho.tolist(),
            "bias_h": temp_b,
            "bias_o": self.bias_o.tolist()
        }

        with open(file_path, "w") as file:
            json.dump(data, file)

    def load(self, file_path):
        """
        Unserializes the weights and biases  produced bi the method "save()".

        Args:
            file_path (str): The path to the file that contains the data
        """
        with open(file_path, "r") as json_file:
            data = json.load(json_file)

            self.weights_ho = np.array(data["weights_ho"], ndmin=2)
            self.bias_o = np.array(data["bias_o"], ndmin=2)

            self.weights = []
            for w in data["weights"]:
                self.weights.append(np.array(w, ndmin=2))

            self.bias_h = []
            for b in data["bias_h"]:
                self.bias_h.append(np.array(b, ndmin=2))


if __name__ == "__main__":
    # in this example the network is being trained to perform as the XOR logical operator

    clss = NeuralNetwork(2, 1, 50, 50)

    data_set = [
        {
            "input": np.array([0, 0], ndmin=2).T,
            "label": np.array([0], ndmin=2).T
        },
        {
            "input": np.array([0, 1], ndmin=2).T,
            "label": np.array([1], ndmin=2).T
        },
        {
            "input": np.array([1, 0], ndmin=2).T,
            "label": np.array([1], ndmin=2).T
        },
        {
            "input": np.array([1, 1], ndmin=2).T,
            "label": np.array([0], ndmin=2).T
        },
    ]

    # for i in range(50000):
    #     current = random.choice(data_set)
    #     clss.train(current["input"], current["label"])

    # clss.save("nn_data.json")

    clss.load("nn_data.json")

    print(clss.predict([1, 1]))
    print(clss.predict([0, 1]))
    print(clss.predict([1, 0]))
    print(clss.predict([0, 0]))
