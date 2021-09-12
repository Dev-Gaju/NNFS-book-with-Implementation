import numpy as np


a = np.array([[2,2,3],
              [2,3,4]])
b = [2,3,4]

d = np.sum(a*b, axis=0)
print("0 axis", d)

d = np.sum(a*b, axis=1)
print("1 axis :", d)

print(-np.log(1-1e-7))