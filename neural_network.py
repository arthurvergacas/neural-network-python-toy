import numpy as np
import random


class NeuralNetwork:
    """
    Neural Network Toy, by Arthur Vergaças.
    Based on Daniel Shiffman's library
    """

    def __init__(self, inputs, hidden, outputs):
        """
        Constructor function
        Current only supports three layers, but leading to change it

        Args:
            inputs (int): number of inputs neurons
            hidden (int): number of hidden neurons
            outputs (int): number of outputs
        """
        self.inputs_num = inputs
        self.hidden_num = hidden
        self.outputs_num = outputs

        self.weights_ih = np.random.rand(self.hidden_num, self.inputs_num)
        self.weights_ho = np.random.rand(self.outputs_num, self.hidden_num)

        self.bias_h = np.random.rand(self.hidden_num, 1)
        self.bias_o = np.random.rand(self.outputs_num, 1)

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
        # input layer -> hidden layer
        hidden_output = self.weights_ih @ inputs
        hidden_output = hidden_output + self.bias_h
        # apply activation function
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

        Args:
            input_array (array): An array of inputs
            label_array (array): An array of labes, according to the inputs
        """

        # create the numpy arrays
        inputs = np.array(input_array, ndmin=2).T
        labels = np.array(label_array, ndmin=2).T

        # FEEDFORWARD ALGORITHM

        # define the hidden layer outputs
        # input layer -> hidden layer
        hidden_output = self.weights_ih @ inputs
        hidden_output = hidden_output + self.bias_h
        # apply activation function
        hidden_output = self.sigmoid(hidden_output)

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
        output_gradient = self.dsig_vec(output_output)  # !!!!
        output_gradient *= output_error
        output_gradient *= self.learning_rate

        # calculate how much the weights must change
        # with the transposed matrix, since it's going backwards
        weights_ho_delta = output_gradient @ hidden_output.T

        # adjust weights
        self.weights_ho += weights_ho_delta
        # adjust bias (its just the gradient)
        self.bias_o += output_gradient

        # hidden layer error
        hidden_error = self.weights_ho.T @ output_error

        # calculate hidden layer gradient (same proccess)
        hidden_gradient = self.dsig_vec(hidden_output)  # !!!!
        hidden_gradient *= hidden_error
        hidden_gradient *= self.learning_rate

        # hidden deltas
        weights_ih_delta = hidden_gradient @ inputs.T

        # adjust weights
        self.weights_ih += weights_ih_delta
        # adjust bias
        self.bias_h += hidden_gradient

    def train_test(self):
        data_set = [
            {
                "input": [0, 0],
                "label": [0]
            },
            {
                "input": [1, 0],
                "label": [1]
            },
            {
                "input": [0, 1],
                "label": [1]
            },
            {
                "input": [1, 1],
                "label": [0]
            },
        ]

        for i in range(42000):
            current = random.choice(data_set)
            self.train(current["input"], current["label"])

        # print(self.predict([1, 0]))


if __name__ == "__main__":
    clss = NeuralNetwork(2, 4, 1)

    data_set = [
        {
            "input": [0, 0],
            "label": [0]
        },
        {
            "input": [1, 0],
            "label": [1]
        },
        {
            "input": [0, 1],
            "label": [1]
        },
        {
            "input": [1, 1],
            "label": [0]
        },
    ]

    for i in range(500000):
        current = random.choice(data_set)
        clss.train(current["input"], current["label"])

    # print(clss.predict([1, 1]))
    # print(clss.predict([0, 1]))
    # print(clss.predict([1, 0]))
    # print(clss.predict([0, 0]))
