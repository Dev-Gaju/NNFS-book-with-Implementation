import Ch6_final as p
import  numpy as np

#dataset
X,y=p.vertical_data(samples=100, classes=3)

#mcode from another model
dens1 = p.Layer_Dense(2,3)

activation1 =p.Activation_ReLU()

dens2 =p.Layer_Dense(3,3)

activation2 = p.Activation_Softmax()

loss_function = p.Loss_CategoricalCrossentropy()



#weight abd bias update procedure
lowest_loss= 9999999
best_dense1_weight =dens1.weights.copy()
best_dense1_bias = dens1.biases.copy()
best_dense2_weight =dens2.weights.copy()
best_dense2_bias = dens2.biases.copy()

# now run a function and check whether weight change or not

for iteration in range(10000):

    # check the formula
    # dens1.weights = 0.05 * np.random.randn(2,3)
    # dens1.biases = 0.05 * np.random.randn(1, 3)
    # dens2.weights =0.05 * np.random.randn(3,3)
    # dens2.biases = 0.05 * np.random.rand(1,3)

    dens1.weights += 0.05 * np.random.randn(2, 3)
    dens1.biases += 0.05 * np.random.randn(1, 3)
    dens2.weights += 0.05 * np.random.randn(3, 3)
    dens2.biases += 0.05 * np.random.randn(1, 3)

    #perform forward pass
    # dens1.forward(X)
    # activation1.forward(dens1.output)
    # dens2.forward(activation1.output)
    # activation2.forward(dens2.output)
    dens1.forward(X)
    activation1.forward(dens1.output)
    dens2.forward(activation1.output)
    activation2.forward(dens2.output)

    #check loss with loss function
    loss = loss_function.calculate(activation2.output, y)

    prediction = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(prediction==y)
    # print(accuracy)
    #update weight and biases
    if loss < lowest_loss:
        print('New set of weight found', iteration, 'loss:',loss, 'accuracy', accuracy)
        best_dense1_weight = dens1.weights.copy()
        best_dense1_bias =dens1.biases.copy()
        best_dense2_weight = dens2.weights.copy()
        best_dense2_bias = dens2.biases.copy()
        lowest_loss = loss
    else:

        dens1.weights = best_dense1_weight.copy()
        dens1.biases = best_dense1_bias.copy()
        dens2.weights = best_dense2_weight.copy()
        dens2.biases = best_dense2_bias.copy()
