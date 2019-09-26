import numpy as np
from numpy import linalg as LA


x = np.random.normal(0,1,(2,4))
print(x)

x += np.linspace(0,1,2).repeat(2)
print(np.linspace(0,1,2).repeat(2))
print(x)

c = np.array([1,2,3,4])
print(x.dot(c))
print((x @ c).shape)

d = np.array([1,2])
print(np.linalg.norm(d))
print(np.linalg.norm(d)**2)
print(d*-1.5)
print(np.exp(d*-1.5))

a = np.arange(9) - 4
b = a.reshape((3, 3))

print(a)
print(b)

print(LA.norm(a, 1))

print(LA.norm(a, axis=0))

a = np.array([2,3,1,0])
print(a)
idx = a.argsort()
print(idx)
print(a[idx])

np.random.normal(0, eps,(1,2))


