import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

                #---Dense layes---
class Dense_layer:
    def __init__(self, n_inputs, n_neurons):
        self.weight = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.bias = 0.01 * np.zeros(1, n_neurons)

    def Forward_pass(self, input):
        self.output =np.dot(input, self.weight) + self.bias

           # ---Activation Function---
'''Relu activation'''
class Relu_Activation :
    def re_Active(self, inputs):
        self.re_output = np.max(0, inputs)

'''Softmax  Activation'''
class Soft_Activation:
    def soft_active(self, input):
        #get unnormalized value
        exponential_values = np.exp(input-np.max(input, axis=1, keepdims=True))
        probabilities =exponential_values/np.sum(exponential_values, axis=True, keepdims=True)
        self.soft_output = probabilities

           # ---Loss Function---
'''class loss'''
class Loss_f:
    # calculate the data anf regularization losses
    #given output and ground truth
    def calculate_losses(self, output, y):
        #calculate the loss
        sample_losse = self.soft_active(output,y)
        data_loss= np.mean(sample_losse)
        return data_loss

'''Cross entropy Function '''
class Categorical_crossentropy(Loss_f):

    def CE_calculate(self, y_pred, y_true):
        #number of losses in a batch
        sample = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        #probabilities for target values

        if len(y_true.shape)==1:
            correct_confidence = y_pred_clipped[range(sample), y_true]
        elif len(y_true.shape)==2:
            correct_confidence = np.sum(y_pred_clipped * y_true, axis=1)

        # Losses
        Negetive_log_likelihood = np.log(correct_confidence)
        return Negetive_log_likelihood



