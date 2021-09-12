import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

                #---Dense layes---
class Dense_layer:
    def __init__(self, n_inputs, n_neurons):
        self.weight = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.bias = 0.01 * np.zeros((1, n_neurons))

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

        ''' 1-1e-7 prevent loss from being negetive makinbg small value insted and greater than 1 and 1e-7 prevent not 0'''

        #probabilities for target values

        if len(y_true.shape)==1:
            correct_confidence = y_pred_clipped[range(sample), y_true]
        elif len(y_true.shape)==2:
            correct_confidence = np.sum(y_pred_clipped * y_true, axis=1)

        # Losses
        Negetive_log_likelihood = -np.log(correct_confidence)
        return Negetive_log_likelihood

X,y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Dense_layer(2,3)
# print(dense1)

# Create ReLU activation (to be used with Dense layer):
activation1 = Relu_Activation()

# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values
dense2 = Dense_layer( 3 , 3 )

# Create Softmax activation (to be used with Dense layer):
Activation_2 = Soft_Activation()

#create a loss function
loss_fuction = Categorical_crossentropy()

# Perform a forward pass of our training data through this layer
dense1.Forward_pass(X)


# Perform a forward pass through activation function
# it takes the output of first dense layer here
activation1.re_Active(dense1.re_output)

# Perform a forward pass through second Dense layer
# it takes outputs of activation function of first layer as inputs
dense2.Forward_pass(activation1.re_output)

# Perform a forward pass through activation function
# it takes the output of second dense layer here
Activation_2.soft_output(dense2.output)

print (Activation_2.soft_output[: 5 ])

loss = loss_fuction.calculate_losses(Activation_2.soft_output, y)
print ( 'loss:' , loss)