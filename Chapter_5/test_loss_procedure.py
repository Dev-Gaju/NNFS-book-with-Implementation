
import numpy as np

            # zip Function
softmax_outputs = np.array([[ 0.7 , 0.1 , 0.2 ],
                            [ 0.1 , 0.5 , 0.4 ],
                            [ 0.02 , 0.9 , 0.08 ]])

class_target = [0,1,1]

for indsx, distribution in  zip(class_target, softmax_outputs):
    print(distribution[indsx])

print(softmax_outputs[[0,1,2], class_target])   #0,1,2 define the row