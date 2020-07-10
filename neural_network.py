import numpy as np
import random as rnd


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

        self.weights_ih = np.full(
            (self.hidden_num, self.inputs_num), rnd.random())
        self.weights_ho = np.full(
            (self.outputs_num, self.hidden_num), rnd.random())

        self.bias_h = np.full((self.hidden_num, 1), rnd.random())
        self.bias_0 = np.full((self.outputs_num, 1), rnd.random())

        self.learning_rate = 0.1

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
            """
            pass

        def train(self, input_array, label_array):
            """
            Trains the neural network with supervised learning
            The 'train' method trains the neural network just once, so it's necessary 
            to train it within a loop

            Args:
                input_array (array): An array of inputs
                label_array (array): An array of labes, according to the inputs
            """
            pass


if __name__ == "__main__":
    clss = NeuralNetwork(2, 3, 1)
