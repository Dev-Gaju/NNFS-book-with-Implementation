import numpy as np
softmax_outputs = np.array([[ 0.7 , 0.1 , 0.2 ],
                            [ 0.1 , 0.5 , 0.4 ],
                            [ 0.02 , 0.9 , 0.08 ]])
class_targets = np.array([[ 1 , 0 , 0 ],
                          [ 0 , 1 , 0 ],
                          [ 0 , 1 , 0 ]])
# Probabilities for target values - # only if categorical labels if len (class_targets.shape) == 1 : correct_confidences = softmax_outputs[ range ( len (softmax_outputs)), class_targets
if len (class_targets.shape) == 1 :   # for [0,1,1]
    correct_confidences = softmax_outputs[ range ( len (softmax_outputs)), class_targets
]
elif len (class_targets.shape) == 2 :  #for upper dimension like 2
    correct_confidences = np.sum( softmax_outputs * class_targets, axis = 1)

neg_log = - np.log(correct_confidences)
average_losses = np.mean(neg_log)    #average loss
print(average_losses)


#check spme