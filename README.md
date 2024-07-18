# Optimization Methods in Deep Learning

This repository contains an investigation of common optimization methods used in deep learning, as part of the coursework for the Deep Learning course by Dr. Soleymani.

## Notebook Content

The notebook `Optimization_Methods_Deep_Learning.ipynb` explores several optimization techniques, providing both theoretical insights and practical implementation examples.

### Content Overview

#### Introduction
The introduction provides an overview of optimization methods in deep learning, explaining their importance in training neural networks effectively and efficiently.

#### Gradient Descent
Gradient Descent is the most basic optimization algorithm. The idea is to update the parameters in the opposite direction of the gradient of the loss function with respect to the parameters. The update rule is given by:

$$
\theta := \theta - \eta \nabla_\theta J(\theta)
$$

where:
- $\theta$ represents the parameters.
- $\eta$ is the learning rate.
- $\nabla_\theta J(\theta)$ is the gradient of the loss function $J$ with respect to $\theta$.

#### Stochastic Gradient Descent (SGD)
SGD is a variant of Gradient Descent where the parameters are updated for each training example. The update rule is:

$$
\theta := \theta - \eta \nabla_\theta J(\theta; x^{(i)}, y^{(i)})
$$

where $x^{(i)}$ and $y^{(i)}$ are the $i$-th training example and label, respectively.

#### Mini-batch Gradient Descent
Mini-batch Gradient Descent is a compromise between Batch Gradient Descent and Stochastic Gradient Descent. The update rule is applied to small batches of training data:

$$
\theta := \theta - \eta \nabla_\theta J(\theta; x^{(i:i+n)}, y^{(i:i+n)})
$$

where $x^{(i:i+n)}$ and $y^{(i:i+n)}$ are batches of training examples and labels.

#### Momentum
Momentum aims to accelerate Gradient Descent by adding a fraction of the previous update to the current update. The update rule is:

$$
\begin{aligned}
v_t = \gamma v_{t-1} + \eta \nabla_\theta J(\theta) \\\
\theta := \theta - v_t
\end{aligned}
$$

where $\gamma$ is the momentum term.

#### Nesterov Accelerated Gradient (NAG)
NAG is a variant of Momentum where the gradient is calculated at the approximate future position of the parameters:

$$
\begin{aligned}
v_t = \gamma v_{t-1} + \eta \nabla_\theta J(\theta - \gamma v_{t-1}) \\\
\theta := \theta - v_t
\end{aligned}
$$

#### Adam Optimizer
Adam (Adaptive Moment Estimation) combines the advantages of two other extensions of stochastic gradient descent. It computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients. The update rules are:

$$
\begin{aligned}
m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla_\theta J(\theta) \\\
v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla_\theta J(\theta))^2 \\\
\hat{m}_t = \frac{m_t}{1 - \beta_1^t} \\\
\hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\\
\theta := \theta - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\endd{aligned}
$$

where:
- $m_t$ and $v_t$ are the first and second moment estimates.
- $\beta_1$ and $\beta_2$ are decay rates for these estimates.
- $\epsilon$ is a small constant to prevent division by zero.

## Results Summary

The notebook provides a detailed comparison of these optimization methods through experiments. Key results include:

- **Gradient Descent**: Simple but can be slow for large datasets.
- **SGD**: Faster convergence compared to Batch Gradient Descent but with more noise in the updates.
- **Mini-batch Gradient Descent**: Balances between the convergence speed of SGD and the stability of Batch Gradient Descent.
- **Momentum**: Accelerates convergence by accumulating a velocity vector in directions of consistent gradient.
- **NAG**: Improves upon Momentum by looking ahead at the future position.
- **Adam**: Generally performs well across various types of neural networks and datasets due to its adaptive learning rate.

## Author
- **Full Name:** Payam Taebi 

## How to Use

1. Clone the repository:
    ```sh
    git clone https://github.com/aqaPayam/optimization-methods-deep-learning.git
    ```
2. Navigate to the repository directory:
    ```sh
    cd optimization-methods-deep-learning
    ```
3. Open the Jupyter Notebook:
    ```sh
    jupyter notebook Optimization_Methods_Deep_Learning.ipynb
    ```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
