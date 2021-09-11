#Relu activation function
import numpy as np

input = [0.3,0.4,0.6,0.3,.49, -.57]
output =[]

for i in input:
    if i > 0:
        output.append(i)
    else:
        output.append(0)

print(output)

'''Second Option'''

outputs = []
for i in input:
    outputs.append(max(0,i))
print("2nd Option of Calculation : ",outputs)

'''Third option'''

output_values=np.maximum(0,input)
print("with Numpy values", output_values)

# !---'''Activation SoftMax'''---!
outputs_value =[4.8, 1.21, 2.385]
E = 2.71
output_ce =[]
for i in outputs_value:
    output_ce.append(E**i)
print(output_ce)
exponeential_sum = sum(output_ce)

output_val =[]

for j in output_ce:
    output_val.append(j/ exponeential_sum)
print("Cross Entropy Values : ", output_val)

'''Lets try about Numpy values'''
outputs =([[1.8, 1.21, 3.385], [2, 1.21, 2.385],[3, 1.21, 2.385]])
exp_val = np.exp(outputs- np.max(outputs, axis=1, keepdims=True))
print("Hey, Um exp_val", exp_val)
final_output = exp_val/np.sum(exp_val, axis=1, keepdims=True)
print("Final output of SoftMax will be : ", final_output)