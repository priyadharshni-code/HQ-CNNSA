!pip install pennylane --upgrade
from IPython.display import clear_output
clear_output(wait=False)

import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt

"""Get the preprocessed input"""

train_data = np.array(pd.read_csv('/content/train.csv'))
test_data = np.array(pd.read_csv('/content/test.csv'))

train_data.shape, test_data.shape

train_len = len(train_data)
test_len = len(test_data)

train_len, test_len

x_train = train_data[:, 0:784]
y_train = train_data[:, 784]

x_test = test_data[:, 0:784]
y_test = test_data[:, 784]

x_train = x_train / 255
x_test = x_test / 255

x_train = x_train.reshape(9931,28,28)
x_test = x_test.reshape(2479,28,28)

"""**Extra dimension is included  for the quanvolutional channels **"""

x_train = np.array(x_train[..., tf.newaxis], requires_grad=True)
x_test = np.array(x_test[..., tf.newaxis], requires_grad=True)

x_train.shape, x_test.shape

"""## Quanvolution Layer - Quantum circuit is applied on 2x2 blocks of (28x28x1) image data, and returns 4 dimensional output."""

n_layers_quanv1 = 1

dev_1 = qml.device("default.qubit", wires=4)
quanv_1_params = np.random.uniform(high=2 * np.pi, size=(n_layers_quanv1, 4))
@qml.qnode(dev_1)
def circuit_1(phi):
       for j in range(4):
        qml.RY(np.pi * phi[j], wires=j)
      RandomLayers(quanv_1_params, wires=list(range(4)))
    return [qml.expval(qml.PauliZ(j)) for j in range(4)]
 phi_values = [0.1, 0.3, 0.5, 0.7]
print(qml.draw(circuit_1)(phi_values))

def quanv_1(image):
    """Convolves the input image with many applications of the same quantum circuit."""
    out = np.zeros((14, 14, 4))


    for j in range(0, 28, 2):
        for k in range(0, 28, 2):

            q_results = circuit_1(
                [
                    image[j, k, 0],
                    image[j, k + 1, 0],
                    image[j + 1, k, 0],
                    image[j + 1, k + 1, 0]
                ]
            )

            for c in range(4):
                out[j // 2, k // 2, c] = q_results[c]
    return out

x_train_quanv_1 = []
x_test_quanv_1 = []


print("Quanvoltional(layer-1) preprocessing of train images:")
for idx, img in enumerate(x_train):
    if((idx+1)%10==0):
        print("Train Image => {}/{}        ".format(idx + 1, train_len))
    x_train_quanv_1.append(quanv_1(img))
x_train_quanv_1 = np.array(x_train_quanv_1)
print("\nQuanvolutional(layer-1) preprocessing of test images:")
for idx, img in enumerate(x_test):
    if((idx+1)%5==0):
        print("Test Image => {}/{}        ".format(idx + 1, test_len))
    x_test_quanv_1.append(quanv_1(img))


x_test_quanv_1 = np.array(x_test_quanv_1)

x_train_quanv_1.shape, x_test_quanv_1.shape

"""Saving it on npy Format"""

np.save("x_train_final.npy", x_train_final)
np.save("y_train.npy", y_train)

np.save("x_test_final.npy", x_test_final)
np.save("y_test.npy", y_test)
