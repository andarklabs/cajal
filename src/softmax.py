import numpy as np

X = np.array([[0.12544906, 0.2544906],
                [0.12333226, 0.14537816],
                [0.11783354, 0.11784958],
                [0.1176308,  0.12707755]])
print(X)
X = X/.2
print(X)
softmax = np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)
print(softmax)
# make an efficient as possible softmax function using shift
shiftx = X - np.max(X, axis=1, keepdims=True)
exp_x = np.exp(shiftx)
sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
softmax = exp_x / sum_exp_x
print(softmax)