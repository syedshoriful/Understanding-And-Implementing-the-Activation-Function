# Understanding-And-Implementing-the-Activation-Function
## Objective
1. To comprehend the conceptual and mathematics underpinning of the Actiavation Function.
2. To execute the Activation Function in a programming language (such as Python).
3. The objective is to examine the attriutes and consequences of using the Activation Function inside neural networks.

Neural networks depend heavily on activation functions, particularly when it comes to deep learning. They give the network non-linearity, which allows it to discover intricate links within the data. We will dig into the realm of activation functions in this post, examining their types, importance, and effects on deep neural network learning.

## What Are Activation Functions?
Activation functions are an integral building block of neural networks that enable them to learn complex patterns in data. They transform the input signal of a node in a neural network into an output signal that is then passed on to the next layer. Without activation functions, neural networks would be restricted to modeling only linear relationships between inputs and outputs.

Activation functions introduce non-linearities, allowing neural networks to learn highly complex mappings between inputs and outputs.

Choosing the right activation function is crucial for training neural networks that generalize well and provide accurate predictions. In this post, we will provide an overview of the most common activation functions, their roles, and how to select suitable activation functions for different use cases.

Whether you are just starting out in deep learning or are a seasoned practitioner, understanding activation functions in depth will build your intuition and improve your application of neural networks.


![image](https://github.com/syedshoriful/Understanding-And-Implementing-the-Activation-Function/assets/94040527/30bf2dab-2b0a-4b3d-8fdd-9994d34073fd)

# Why Are Activation Functions Essential?
Without activation functions, neural networks would just consist of linear operations like matrix multiplication. All layers would perform linear transformations of the input, and no non-linearities would be introduced.

Most real-world data is non-linear. For example, relationships between house prices and size, income, and purchases, etc., are non-linear. If neural networks had no activation functions, they would fail to learn the complex non-linear patterns that exist in real-world data.

Activation functions enable neural networks to learn these non-linear relationships by introducing non-linear behaviors through activation functions. This greatly increases the flexibility and power of neural networks to model complex and nuanced data.

# Types of Activation Functions
Sigmoid Function: The sigmoid function is a type of logistic function that maps input values to a range between 0 and 1. It's often used to produce probabilities in binary classification problems. The formula for the sigmoid function is:
![image](https://github.com/syedshoriful/Understanding-And-Implementing-the-Activation-Function/assets/94040527/1508466e-48a0-462b-b827-b367454c5e45)


