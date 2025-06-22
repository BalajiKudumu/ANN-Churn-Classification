# ANN-Churn-Classification Project

## Artificial Neural Network (ANN) for Classification

## Overview
An **Artificial Neural Network (ANN)** is a computational model inspired by biological neurons that can learn complex mappings between inputs and outputs.  
This README outlines the architecture and working of an ANN for a typical **classification** problem.

---

## ANN Architecture
A typical feed-forward ANN for classification consists of three types of layers:

1. **Input Layer**  
   - Represents the features of the dataset.  
   - Number of nodes = number of input features.

2. **Hidden Layers** (one or more)  
   - Perform most of the computation.
   - Each neuron computes a weighted sum of its inputs and applies a nonlinear **activation function** like `ReLU`, `Sigmoid`, or `Tanh`.
   - Number of hidden layers and neurons per layer depends on problem complexity.

3. **Output Layer**  
   - Produces the final class predictions.
   - Number of output nodes:
     - Binary classification → 1 neuron with `Sigmoid` activation.
     - Multi-class classification → *n* neurons with `Softmax` activation.

---

## Mathematical Explanation
Each neuron in a layer computes:

$$
\text{Output} = f\left(\sum_{i=1}^n w_i x_i + b\right)
$$

Where:
- $x_i$ = Input feature or previous layer’s neuron output
- $w_i$ = Trainable weight
- $b$ = Bias
- $f$ = Activation function (e.g. ReLU, Sigmoid, Softmax)

---

## Example Architecture (Binary Classification)
| Layer            | Neurons | Activation |
|------------------|---------|------------|
| Input            | 12      | —          |
| Hidden Layer 1   | 64      | ReLU       |
| Hidden Layer 2   | 32       | ReLU       |
| Output           | 1       | Sigmoid    |

---

## Training Process
1. **Loss function**:
   - Binary classification → `Binary Cross-Entropy`
   - Multi-class classification → `Categorical Cross-Entropy`

2. **Optimizer**:  
   Popular choices include `SGD`, `Adam`, and `RMSProp`.

3. **Training Steps**:
   - **Forward propagation**: Predict output given inputs.
   - **Loss calculation**: Measure error against ground truth.
   - **Backpropagation**: Adjust weights to reduce the error.

---
This repository contains a Python implementation of an Artificial Neural Network (ANN) for customer churn prediction. The ANN is built using TensorFlow/Keras and provides an end-to-end solution for binary classification tasks on tabular data.

---
## Example in Python (Keras)
Here’s a quick code snippet for a binary classification model using Keras:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(16, activation='relu', input_shape=(10,)),  # 12 features
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')                    # Binary output
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
