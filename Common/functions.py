import numpy as np

# 항등함수
def identity_function(x):
    return x

# 계단함수
def step_function(x):
    return np.array(x > 0, dtype=np.int)

# 시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

# 시그모이드 함수
def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
    
# ReLU 함수
def relu(x):
    return np.maximum(0, x)

# ReLU 함수
def relu_grad(x):
    grad = np.zeros_like(x)
    grad[x>=0] = 1
    return grad
    
# 소프트맥스 함수
def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

# SSE 함수
def sum_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

# CSE 함수
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

# 소프트맥스 손실함수
def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)