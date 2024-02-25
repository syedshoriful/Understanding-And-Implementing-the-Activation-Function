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

### Sigmoid Function: 
The sigmoid function is a type of logistic function that maps input values to a range between 0 and 1. It's often used to produce probabilities in binary classification problems. The formula for the sigmoid function is:

![image](https://github.com/syedshoriful/Understanding-And-Implementing-the-Activation-Function/assets/94040527/554f0bbd-9094-4259-8522-596092130d95)

![image](https://github.com/syedshoriful/Understanding-And-Implementing-the-Activation-Function/assets/94040527/fed514d6-3e7a-43cf-b0ef-4b3cd5d41889)

### Tanh Function:
The hyperbolic tangent function, or tanh, is similar to the sigmoid function but maps input values to a range between -1 and 1. It is symmetric around the origin and is commonly used in neural networks. The formula for the tanh function is:

![image](https://github.com/syedshoriful/Understanding-And-Implementing-the-Activation-Function/assets/94040527/2b723e27-a05b-4f28-b5f4-674721764a55)

![image](https://github.com/syedshoriful/Understanding-And-Implementing-the-Activation-Function/assets/94040527/077f7ce8-b46c-44b0-b375-af02fe2c8b42)

### Rectified Linear Unit (ReLU): 
ReLU is a simple and widely used activation function that outputs the input directly if it is positive, and zero otherwise. It has been found to accelerate the convergence of stochastic gradient descent during training. The formula for ReLU is:

![image](https://github.com/syedshoriful/Understanding-And-Implementing-the-Activation-Function/assets/94040527/876081cb-bd5b-4680-abe5-589b883d5737)

![image](https://github.com/syedshoriful/Understanding-And-Implementing-the-Activation-Function/assets/94040527/57003530-03bd-4b52-97a1-b5188cd069ed)

### Leaky ReLU:
Leaky ReLU is a variation of the ReLU activation function that allows a small, positive gradient when the input is negative, which helps address the "dying ReLU" problem where neurons may become inactive during training. The formula is:

![image](https://github.com/syedshoriful/Understanding-And-Implementing-the-Activation-Function/assets/94040527/54482528-463d-4079-b878-5da4c8d2e0b6)

![image](https://github.com/syedshoriful/Understanding-And-Implementing-the-Activation-Function/assets/94040527/c2037366-bffa-46b3-805b-bcbe222a00cc)


### Parametric ReLU: 
Parametric ReLU (PReLU) is similar to Leaky ReLU, but instead of using a fixed slope for negative inputs, it allows the slope to be learned during training.

![image](https://github.com/syedshoriful/Understanding-And-Implementing-the-Activation-Function/assets/94040527/8a167616-3091-4cd0-a608-e806a98de820)

![image](https://github.com/syedshoriful/Understanding-And-Implementing-the-Activation-Function/assets/94040527/47e13192-6c12-4582-a005-4afd0ec2fac6)


### Exponential Linear Unit (ELU): 
ELU is an activation function that smoothly saturates to a negative value for inputs less than zero, which helps alleviate the vanishing gradient problem. The formula for ELU is:

![image](https://github.com/syedshoriful/Understanding-And-Implementing-the-Activation-Function/assets/94040527/79280a91-0b57-4bf4-a03e-04ec6056babf)

![image](https://github.com/syedshoriful/Understanding-And-Implementing-the-Activation-Function/assets/94040527/c9e83133-c1ba-4743-85d7-4705b91648b8)



### Scaled Exponential Linear Unit (SELU): 
SELU is a variation of ELU that introduces a self-normalizing property, meaning that the output of each layer will preserve a mean of 0 and a standard deviation of 1, which can help stabilize training in deep neural networks.

![image](https://github.com/syedshoriful/Understanding-And-Implementing-the-Activation-Function/assets/94040527/2b175f09-a48f-4a4e-9501-942ba4b111b1)

![image](https://github.com/syedshoriful/Understanding-And-Implementing-the-Activation-Function/assets/94040527/2aaba150-2931-416d-8e33-503d06f9f60c)



### Softmax Function:
Softmax Function is used output layers for multi-class classification. Output probabilities for each class, summing to 1. This activation function ensures valid probability distribution for predictions.

![image](https://github.com/syedshoriful/Understanding-And-Implementing-the-Activation-Function/assets/94040527/92540ca4-389b-47a2-abc2-91854b825c61)

![image](https://github.com/syedshoriful/Understanding-And-Implementing-the-Activation-Function/assets/94040527/d7863f08-1494-4068-ab18-0c9f35dbfdc0)

## Calculate the derivative of the Activation Function and explain it's significance in the backpropgation process.
Let's take the derivative of the ReLU activation function as an example, since it's one of the most commonly used activation functions. The ReLU function is defined as:

![image](https://github.com/syedshoriful/Understanding-And-Implementing-the-Activation-Function/assets/94040527/4827dc43-fdf2-4cf4-a570-17ba3118d62b)


To calculate its derivative, we need to consider its piecewise nature:

![image](https://github.com/syedshoriful/Understanding-And-Implementing-the-Activation-Function/assets/94040527/5eb0df67-9fc2-4a6d-aea9-46eba0ea9bf0)


This derivative indicates that if the input to the ReLU function is positive, the derivative is 1, and if the input is negative or zero, the derivative is 0.

The significance of the derivative in the backpropagation process lies in the computation of gradients used to update the weights of the neural network during training. Here's how it works:

1. **Forward Pass**:
   - During the forward pass, the activations of each neuron are computed using the ReLU function.
   - These activations are then fed forward through the network to compute the output.

2. **Backward Pass (Backpropagation)**:
   - In the backward pass, the gradient of the loss function with respect to the output of the neuron is computed.
   - This gradient is then backpropagated through the network to compute the gradients of the loss function with respect to the weights of each neuron.
   - The chain rule of calculus is used to compute these gradients, where the derivative of the activation function plays a crucial role.
   - For neurons that use ReLU activation, if the input was positive during the forward pass (i.e., \(x > 0\)), the gradient is preserved (i.e., \(f'(x) = 1\)). This means that the gradient flows through the neuron unaffected.
   - However, if the input was negative or zero during the forward pass (i.e., \(x ≤ 0\)), the gradient becomes 0. This effectively stops the gradient flow through the neuron, preventing any updates to the weights of that neuron. This property is what helps mitigate the vanishing gradient problem, as it prevents gradients from becoming too small.

In summary, the derivative of the activation function is crucial in backpropagation for computing gradients, which are used to update the weights of the neural network. For ReLU and similar activation functions, the derivative controls the flow of gradients during backpropagation, allowing effective training by preventing the vanishing gradient problem.

## References
https://www.datacamp.com/tutorial/introduction-to-activation-functions-in-neural-networks
https://medium.com/@sruthy.sn91/understanding-activation-functions-in-deep-learning-1e3728eeaee
https://ml-explained.com/blog/activation-functions-explained












