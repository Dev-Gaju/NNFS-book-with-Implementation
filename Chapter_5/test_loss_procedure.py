
import numpy as np

            # zip Function
softmax_outputs = np.array([[ 0.7 , 0.1 , 0.2 ],
                            [ 0.1 , 0.5 , 0.4 ],
                            [ 0.02 , 0.9 , 0.08 ]])
class_target = [0,1,1]

print("Length : ",len(softmax_outputs))
print("Shape : ",softmax_outputs.shape)

for indx, distribution in  zip(class_target, softmax_outputs):
    print(distribution[indx])

print(softmax_outputs[[0,1,2], class_target])   #0,1,2 define the row
b=softmax_outputs[range(len(softmax_outputs)), class_target]
print("here We are", b)  #size of array can also work
# print(softmax_outputs[range(0,3), class_target])

''' Now apply Cross entropy on this values'''
print("Final CE", -np.log(softmax_outputs[range(len(softmax_outputs)), class_target]))

cal = -np.log(softmax_outputs[range(len(softmax_outputs)), class_target])
average_loss = np.mean(cal)
print(average_loss)


''' What if class target have 3*3 dimension instead of 1'''
softmax_outputs = np.array([[ 0.7 , 0.1 , 0.2 ],
                            [ 0.1 , 0.5 , 0.4 ],
                            [ 0.02 , 0.9 , 0.08 ]])

class_target = np.array([[1,0,0],
                         [0,1,0],
                         [0,1,0]])

output = np.sum(softmax_outputs*class_target, axis=1)
print('average_loss', np.mean(-np.log(output)))
# print(output)