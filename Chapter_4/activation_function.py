import numpy as np
#!----Relu activation function-----!

inputs = [1, 2, -2, 100, -130, 22, 0, 1, 4]
outputs =[]
for i in inputs:
    if i > 0:
        outputs.append(i)
    else:
        outputs.append(0)

print(outputs)

# print( np.exp(3))
# print(2.718**3)



#another way
inputs = [1, 2,-2, 100, -130,22,0,1,4]
outputs =[]
for i in inputs:
        outputs.append(max(0,i))
print(outputs)


#another way using numpy
outputs=np.maximum(0,inputs)
print(outputs)



#                 !-----soft_max activation function------!
outputs =[4.8, 1.21, 2.385]
E = 2.71828
exp_values = []
for values in outputs:
    exp_values.append(E**values)

print(exp_values)


norm_base = sum(exp_values)

norm_values = []

for n_value in exp_values:
    norm_values.append(n_value / norm_base)

print("CE", norm_values)


#Do it with numpy
outputs =([[1.8, 1.21, 3.385], [2, 1.21, 2.385],[3, 1.21, 2.385]])
exp_values = np.exp(outputs)
probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
print("Hello", probabilities)

np.exp(1000)  #doesn't take maximum number


# for this reason write code as

exp_values = np.exp(outputs-np.max(outputs, axis=1, keepdims=True))
print("Exponential values",exp_values)
probabilities = exp_values /np.sum(exp_values, axis =1, keepdims=True)
print("Hi", probabilities)


    