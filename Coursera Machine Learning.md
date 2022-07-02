# Machine Learning

Course website: [Machine Learning | Coursera](https://www.coursera.org/learn/machine-learning).



## 1 - Introduction

Supervised Learning: Given "right answers" to train.

Unsupervised Learning: No "right answer".

Regression: Predict continuous-valued output.

Classification: Predict discrete-valued output.

### 1.1 - Linear Regression with One Variable

| Parameters    | $\theta_0,\theta_1$                                          |
| :------------ | :----------------------------------------------------------- |
| Hypothesis    | $h_\theta(x)=\theta_0+\theta_1x$                             |
| Cost Function | $J(\theta_0,\theta_1)=\frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2\\$ |
| Goal          | minimize $J(\theta_0,\theta_1)$                              |

### 1.2 - Gradient Descent

Update until converge for $\theta_0,\theta_1$.

$\theta_0:=\theta_0-\alpha\frac{\partial}{\partial\theta_0}J(\theta_0,\theta_1)\\$

$\theta_1:=\theta_1-\alpha\frac{\partial}{\partial\theta_1}J(\theta_0,\theta_1)\\$



## 2 - Linear Regression with Multiple Variables

| Parameters       | $\theta_0,\theta_1,...,\theta_n$                             |
| ---------------- | ------------------------------------------------------------ |
| Hypothesis       | $h_\theta(x)=\theta_0+\theta_1x_1+\dots+\theta_jx_j$         |
| Cost Function    | $J(\theta)=\frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2\\$ |
| Goal             | minimize $J(\theta_j)$                                       |
| Gradient Descent | $\theta_j:=\theta_j-\alpha\frac{\partial}{\partial\theta_j}J(\theta)\\$ |

### 2.1 - Feature Scaling

Feature Scaling: Get every feature into approximately $-1\le{x_i}\le1$ range.

$x_i:=\frac{x_i}{\text{max}(|x_i|)}\\$

Mean Normalization: Make features have approximately $0$ mean.

$x_i:=\frac{x_i-\mu_i}{\sigma_i}\\$

$x_i:=\frac{x_i-\mu_i}{\text{max}(x_i)-\text{min}(x_i)}\\$

### 2.2 - Learning Rate $\alpha$

$\alpha$ too small: Slow convergence.

$\alpha$ too large: $J(\theta)$ may increase on every iteration, may not converge.

Choosing $\alpha$ : $\dots0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1\dots$

### 2.3 - Normal Equation

Intuition: $\frac{\partial}{\partial\theta_1}J(\theta)=\frac{\partial}{\partial\theta_2}J(\theta)=\dots=\frac{\partial}{\partial\theta_j}J(\theta)=0\\$

Solution: $\theta=(X^TX)^{-1}X^Ty$

Example: $m$ training examples, $n=4$ features.

$ \theta=\begin{bmatrix}\theta_0\\\theta_1\\\theta_2\\\theta_3\\\theta_4\end{bmatrix}$ $X=\begin{bmatrix}1&x_1^{(1)}&x_2^{(1)}&x_3^{(1)}&x_4^{(1)}\\1&x_1^{(2)}&x_2^{(2)}&x_3^{(2)}&x_4^{(2)}\\1&x_1^{(3)}&x_2^{(3)}&x_3^{(3)}&x_4^{(3)}\\\vdots&\vdots&\vdots&\vdots&\vdots\\1&x_1^{(m)}&x_2^{(m)}&x_3^{(m)}&x_4^{(m)}\end{bmatrix}$ $Y=\begin{bmatrix}y^{(1)}\\y^{(2)}\\y^{(3)}\\\vdots\\y^{(m)}\end{bmatrix}$

Note that, $x_1^{(3)}$ denotes the 1st attribute of the 3rd training example.

| Gradient Descent                    | Normal Equation                                              |
| ----------------------------------- | ------------------------------------------------------------ |
| Need to choose $\alpha$.            | No need to choose $\alpha$.                                  |
| Need many iterations to converge.   | No need to iterate.                                          |
| Still works well when $n$ is large. | Need to compute $(X^TX)^{-1}$ with complexity $O(n^3)$.<br>Slow if $n$ is large (over 100000). |



## 3 - Classification

### 3.1 - Logistic Regression

#### Hypothesis

Sigmoid/Logistic Function:  $g(x)=\frac{1}{1+e^{-x}}\\$

$h_\theta(x)=g(\theta^Tx)=\frac{1}{1+e^{-\theta^Tx}}\\$ , which denotes the possibility of $x$ as class 1.

#### Decision Boundary

  $\begin{aligned}&y=1\\\Leftrightarrow\ &P(y=1|x;\theta)=h_\theta(x)=g(\theta^Tx)\ge0.5\\\Leftrightarrow \ &\theta^Tx\ge0\end{aligned}$

#### Non-linear Decision Boundary

Add polynomial features like $x_1^2, x_2^2$ so that to model can fit more complex examples.

#### Cost Function

| Minus Log Cost                               | Mean Square Cost                                            |
| ------------------------------------------------------------ | ----------------------------------- |
| $\text{Cost}(h_\theta(x),y)=\begin{cases}-\log(h_\theta(x))\ ,y=1\\-\log(1-h_\theta(x))\ ,y=0\end{cases}$ | $\text{Cost}(h_\theta(x),y)=\frac{1}{2}(h_\theta(x)-y)^2\\$ |
| Have a unique global minimum point. (convex)                 | Have many local minimum points. (non-convex)                |

$\begin{aligned}J(\theta)&=\frac{1}{m}\sum_{i=1}^m\text{Cost}(h_\theta(x^{(i)}),y^{(i)})\\&=-\frac{1}{m}\sum_{i=1}^m\bigg[y^{(i)}\text{log}(h_\theta(x^{(i)}))+(1-y^{(i)})\text{log}(1-h_\theta(x^{(i)}))\bigg]\end{aligned}$

### 3.2 - Advanced Optimization

| Examples                             | Advantages                                                   |
| ------------------------------------ | ------------------------------------------------------------ |
| Conjugate Gradient<br>BFGS<br>L-BFGS | No need to pick $\alpha$. <br>Often faster than gradient descent. |

### 3.3 - Multiclass Classification: One-vs-all

For classification problem with $n>2$ categories, we have $y=\{0,1,2,\dots,n-1\}$ .

Train a logistic regression classifier $h_\theta^{(i)}(x)$ for each class $y=i$ to predict the probability $P(y=i|x;\theta)$.

To make a prediction on a new $x$, pick the class that maximizes $h_\theta(x)$.

### 3.4 - Solving the Problem of Overfitting

| Underfit  |  Well-fit  |    Overfit    |
| :-------: | :--------: | :-----------: |
| High Bias | Just Right | High Variance |

There are two main options to address the issue of overfitting:

1) Reduce the number of features:

- Manually select which features to keep.
- Use a model selection algorithm.

2) Regularization:

- Keep all the features, but reduce the magnitude of parameters $\theta_j$.
- Regularization works well when we have a lot of slightly useful features.

#### Cost Function

Add a penalty term to eliminate the parameters. ($\theta_0$ not included)

$J(\theta)=\frac{1}{m}\sum_{i=1}^m\text{Cost}(h_\theta(x^{(i)}),y^{(i)})+\lambda\sum_{j=1}^n\theta_j^2\\$

Regularization Parameter $\lambda$ too large: Cause underfitting.

#### Regularized Linear Regression

| Gradient Descent                                             | Normal Equation                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| $\begin{aligned}\theta_0&:=\theta_0-\alpha\frac{\partial}{\partial\theta_0}J(\theta)\\\theta_j&:=\theta_j-\alpha\bigg[\frac{\partial}{\partial\theta_j}J(\theta)+\frac{\lambda}{m}\theta_j\bigg]\\&:=\big(1-\frac{\lambda}{m}\big)\theta_j-\alpha\frac{\partial}{\partial\theta_j}J(\theta)\space,j\in\{1,2,\dots,n\}\end{aligned}$ | $\theta=(X^TX+\lambda L)^{-1}X^Ty$<br>where $L=\begin{bmatrix}0&0&0&\cdots&0\\0&1&0&\cdots&0\\0&0&1&\cdots&0\\\vdots&\vdots&\vdots&\ddots&\vdots\\0&0&0&\cdots&1\end{bmatrix}$ |

#### Regularized Logistic Regression

$\theta_0:=\theta_0-\alpha\frac{\partial}{\partial\theta_0}J(\theta)\\$

$\begin{aligned}\theta_j&:=\theta_j-\alpha\bigg[\frac{\partial}{\partial\theta_j}J(\theta)+\frac{\lambda}{m}\theta_j\bigg]\\&:=\big(1-\frac{\lambda}{m}\big)\theta_j-\alpha\frac{\partial}{\partial\theta_j}J(\theta)\space,j\in\{1,2,...n\}\end{aligned}$



## 4 - Neural Networks: Representation

### 4.1 - Model Representation

| First Layer | Intermediate Layer | Last Layer   |
| ----------- | ------------------ | ------------ |
| Input Layer | Hidden Layer       | Output Layer |

$g(x)$ : activation function

$a_i^{(j)}$ : "activation" of unit $i$ in layer $j$

$\Theta^{(j)}$ : matrix of weights controlling function mapping from layer $j$ to layer $j+1$

<img src="pics\4-example_network.png" alt="example_network" style="zoom:50%;" />

In the example network above, the data flow looks like:

$[x_1,x_2,x_3]\rightarrow\big[a_1^{(2)},a_2^{(2)},a_3^{(2)}\big]\rightarrow h_\theta(x)\\$

The values for each of the "activation" nodes is obtained as follows:

$a_1^{(2)}=g\big(\Theta_{10}^{(1)}x_0+\Theta_{11}^{(1)}x_1+\Theta_{12}^{(1)}x_2+\Theta_{13}^{(1)}x_3\big)\\$

$a_2^{(2)}=g\big(\Theta_{20}^{(1)}x_0+\Theta_{21}^{(1)}x_1+\Theta_{22}^{(1)}x_2+\Theta_{23}^{(1)}x_3\big)\\$

$a_3^{(2)}=g\big(\Theta_{30}^{(1)}x_0+\Theta_{31}^{(1)}x_1+\Theta_{32}^{(1)}x_2+\Theta_{33}^{(1)}x_3\big)\\$

$h_\theta(x)=a_1^{(3)}=g\big(\Theta_{10}^{(2)}a_0^{(2)}+\Theta_{11}^{(2)}a_1^{(2)}+\Theta_{12}^{(2)}a_2^{(2)}+\Theta_{13}^{(2)}a_3^{(2)}\big)\\$

Where the addition $\Theta^{(j)}$ is the "bias", whose corresponding inputs $x_0,a_0^{(2)}=1$.

The matrix representation of the above computations is like:

| Inputs                                                       | Weights                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| $a^{(1)}=x=\begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix}\overset{\text{add bias}}{\Longrightarrow}\begin{bmatrix}x_0\\x_1\\x_2\\x_3\end{bmatrix}$ | $\Theta^{(1)}=\begin{bmatrix}\Theta_{10}^{(1)}&\Theta_{11}^{(1)}&\Theta_{12}^{(1)}&\Theta_{13}^{(1)}\\\Theta_{20}^{(1)}&\Theta_{21}^{(1)}&\Theta_{22}^{(1)}&\Theta_{23}^{(1)}\\\Theta_{30}^{(1)}&\Theta_{31}^{(1)}&\Theta_{32}^{(1)}&\Theta_{33}^{(1)}\end{bmatrix}$ |
| $a^{(2)}=g\big(z^{(2)}\big)=g\big(\Theta^{(1)}a^{(1)}\big)=\begin{bmatrix}a_1^{(2)}\\a_2^{(2)}\\a_3^{(2)}\end{bmatrix}\overset{\text{add bias}}{\Longrightarrow}\begin{bmatrix}a_0^{(2)}\\a_1^{(2)}\\a_2^{(2)}\\a_3^{(2)}\end{bmatrix}$ | $\Theta^{(2)}=\begin{bmatrix}\Theta_{10}^{(2)}&\Theta_{11}^{(2)}&\Theta_{12}^{(2)}&\Theta_{13}^{(1)}\end{bmatrix}$ |
| $a^{(3)}=g\big(z^{(3)}\big)=g\big(\Theta^{(2)}a^{(2)}\big)=\begin{bmatrix}a_1^{(3)}\end{bmatrix}$ | none                                                         |

### 4.2 - Multiclass Classification

One-vs-all for neural networks: Define the set of resulting classes like:

$y^{(i)}=\begin{bmatrix}1\\0\\0\\0\end{bmatrix},\begin{bmatrix}0\\1\\0\\0\end{bmatrix},\begin{bmatrix}0\\0\\1\\0\end{bmatrix},\begin{bmatrix}0\\0\\0\\1\end{bmatrix}$



## 5 - Neural Networks: Learning

### 5.1 - Cost Function

Take the multiclass classification problem as example:

**Logistic Regression:** $J(\theta)=-\frac{1}{m}\sum_{i=1}^m\bigg[y^{(i)}\text{log}(h_\theta(x^{(i)}))+(1-y^{(i)})\text{log}(1-h_\theta(x^{(i)}))\bigg]+\frac{\lambda}{2m}\sum_{j=1}^n\theta_j^2\\$

**Neural Network:** $J(\theta)=-\frac{1}{m}\sum_{i=1}^m\sum_{k=1}^K\bigg[y^{(i)}\text{log}(h_\theta(x^{(i)})_k)+(1-y^{(i)})\text{log}(1-h_\theta(x^{(i)})_k)\bigg]+\frac{\lambda}{2m}\sum_{l=1}^{L-1}\sum_{i=1}^{s_l}\sum_{j=1}^{s_l+1}\big(\Theta_{j,i}^{(l)}\big)^2\\$

$L$ : total number of layers in the network 

$s_l$ : number of units (not counting bias unit) in layer $l$

$K$ : number of output units/classes

**The first part:** Sum up all the costs for each output class ($K$ in total).

**The second part:** Sum up the squared values of all parameters except bias.

### 5.2 - Backpropagation

[反向传播算法 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/40761721)

[Backpropagation Algorithm | Coursera](https://www.coursera.org/learn/machine-learning/supplement/pjdBA/backpropagation-algorithm)

#### **Problem Setting**

Our object is to calculate $\frac{\partial J(\Theta)}{\partial\Theta_{i,j}^{(l)}}\\$ for all $\Theta_{i,j}^{(l)}$ .

$l$ : index of the layer

$i$ : index of the neuron in layer $l$

$j$ : index of the neuron in layer $l+1$

$\Theta_{i,j}^{(l)}$ : weight from the $i$ th neuron in layer $l$ to the $j$ th neuron in layer $l+1$

#### Preliminary Deduction

It is quite hard to calculate $\frac{\partial J(\Theta)}{\partial\Theta_{i,j}^{(l)}}\\$ directly.

By applying the chain rule of derivatives, we get $\frac{\partial J(\Theta)}{\partial\Theta_{i,j}^{(l)}}=\frac{\partial J(\Theta)}{\partial z_j^{(l+1)}}\frac{\partial z_j^{(l+1)}}{\partial\Theta_{i,j}^{(l)}}\\$ .

It is easy to deduce that $\frac{\partial z_j^{(l+1)}}{\partial\Theta_{i,j}^{(l)}}=\frac{\partial\bigg(\sum_{k=0}^{N(l)}\Theta_{k,j}^{(l)}a_k^{(l)}\bigg)}{\partial\Theta_{i,j}^{(l)}}=a_i^{(l)}\\$ . 

So we define "error value" $\delta_j^{(l+1)}=\frac{\partial J(\Theta)}{\partial z_j^{(l+1)}}\\$ , and then we have $\frac{\partial J(\Theta)}{\partial\Theta_{i,j}^{(l)}}=\delta_j^{(l+1)}a_i^{(l)}\\$ .

Since we already got $a_i^{(l)}$, our object is to calculate $\delta_j^{(l+1)}$ for all $j,l$ .

#### Algorithm Summary

1. Perform forward propagation to compute $a^{(l)}$ for $l=1,2,3,...,L$ .

2. Compute $\delta_j^{(L)}=a^{(L)}-y^{(t)}$ .

3. Compute $\delta_j^{(L-1)},\delta_j^{(L-2)},\dots,\delta_j^{(2)}$ , using $\delta^{(l)}=\big(\Theta^{(l)}\big)^T\delta^{(l+1)}\times g'\big(z^{(l)}\big)$ .

4. Compute $\Delta^{(l)}_{i,j}:=\delta_i^{(l+1)}a_j^{(l)}$ , or with vectorization, $\Delta^{(l)}:=\delta^{(l+1)}\big(a^{(l)}\big)^T$ .

5. If we didn't apply regularization, then $D_{i,j}^{(l)}=\frac{\partial J(\Theta)}{\partial\Theta_{i,j}^{(l)}}=\frac{1}{m}\Delta^{(l)}_{i,j}\\$ .

   Otherwise, We have $D_{i,j}^{(l)}=\frac{\partial J(\Theta)}{\partial\Theta_{i,j}^{(l)}}=\begin{cases}\frac{1}{m}\big(\Delta^{(l)}_{i,j}+\lambda\Theta_{i,j}^{(l)}\big)\space,j\ne0\\\frac{1}{m}\Delta^{(l)}_{i,j}\space,j=0\end{cases}$ .

### 5.3 - Random Initialization

Initialize each $\Theta_{i,j}^{(l)}$ to a same number: Bad! All the units will compute the same thing, giving a highly redundant representation.

Initialize each $\Theta_{i,j}^{(l)}$ to a random value in $[-\epsilon,\epsilon]$ : Good! This breaks the symmetry and helps our network learn something useful.

### 5.4 - Training a Neural Network

1. Randomly initialize the weights.
2. Implement forward propagation to get $h_\Theta(x^{(i)})$ for any $x^{(i)}$ .
3. Implement the cost function.
4. Implement backpropagation to compute partial derivatives.
5. Use gradient checking to confirm that your backpropagation works. Then disable gradient checking.
6. Use gradient descent or a built-in optimization function to minimize the cost function with the weights in theta.



## 6 - Advice for Applying Machine Learning

### 6.1 - Evaluating a Learning Algorithm

Once we have done some trouble shooting for errors in our predictions by: 

- Getting more training examples.
- Trying smaller sets of features.
- Trying additional features.
- Trying polynomial features.
- Increasing or decreasing $\lambda$ .

One way to break down our dataset into the three sets is:

- Training set: 60%.
- Cross validation set: 20%.
- Test set: 20%.

We can move on to evaluate our new hypothesis:

1. Learn $\Theta$ and minimize $J_{train}(\Theta)$ .
2. Find the best model according to $J_{cv}(\Theta)$ .
3. Compute the test set error $J_{test}(\Theta)$ .

### 6.2 - Bias vs. Variance

#### Checking Model Complexity

<img src="pics\6-parameter.png" style="zoom:50%;" />

|                         | Training Set Error | Cross Validation Set Error |
| ----------------------- | ------------------ | -------------------------- |
| Underfit (High Bias)    | High               | High                       |
| Overfit (High Variance) | Low                | High                       |

#### Regularization

<img src="pics\6-lambda.png" style="zoom:100%;" />

|                                     | Training Set Cost | Cross Validation Set Cost |
| ----------------------------------- | ----------------- | ------------------------- |
| $\lambda$ too large (High Bias)     | High              | High                      |
| $\lambda$ too small (High Variance) | Low               | High                      |

#### Learning Curves

<img src="pics\6-learning_curve1.png"  style="zoom:75%;" /><img src="pics\6-learning_curve2.png"  style="zoom:75%;" />

**Experiencing High Bias:**

Model underfits the training set and cross validation set, getting more training data will not help much.

**Experiencing High Variance:**

Model overfits the training set, getting more training data is likely to help.

### 6.3 - Deciding What to Do Next

- **Getting more training examples:** Fixes high variance

- **Trying smaller sets of features:** Fixes high variance

- **Adding features:** Fixes high bias

- **Adding polynomial features:** Fixes high bias

- **Decreasing $\lambda$:** Fixes high bias

- **Increasing $\lambda$:** Fixes high variance.

### 6.4 - Error Analysis ( Quite Practical and Useful !!!!!!!!!! ) 

- Start with a simple algorithm, implement it quickly, and test it early on your cross validation data.
- Plot learning curves to decide if more data, more features, etc. are likely to help.
- Manually examine the errors on examples in the cross validation set and try to spot a trend where most of the errors were made.
- Make sure the quick implementation incorporated a single real number evaluation metric.

### 6.5 - Handling Skewed Data

To better handle skewed data, we can use precision and recall to evaluate the model.

$\text{precision}=\frac{\text{true positive}}{\text{true positive + false positive}}\\$

$\text{recall}=\frac{\text{true positive}}{\text{true positive + false negative}}\\$

We can further give a trade off between precision and recall using what is called F1 score.

$\text{F1 score}=2\frac{\text{precision}\times\text{recall}}{\text{precision + recall}}\\$



## 7 - Support Vector Machines

### 7.1 - SVM Hypothesis

Just like liner regression, we gain the cost function of SVM:

$J(\theta)=C\sum_{i=1}^{m}\big[y^{(i)}\text{cost}_1(\Theta^Tx^{(i)})+(1-y^{(i)})\text{cost}_0(\Theta^Tx^{(i)})\big]+\frac{1}{2}\sum_{j=1}^{n}\theta_j^2\\$

The cost function for each label looks like:

<img src="pics\7-cost.png" style="zoom:50%;" />

So our hypothesis is:

$h_\theta(x)=\begin{cases}1\space,\Theta^Tx\ge1\\0\space,\Theta^Tx\le-1\end{cases}$

Given a trained weight $\Theta$ , we want each example to be classified correctly, so we have:

$\sum_{i=1}^{m}\big[y^{(i)}\text{cost}_1(\Theta^Tx^{(i)})+(1-y^{(i)})\text{cost}_0(\Theta^Tx^{(i)})\big]=0\\$

$\begin{aligned}\text{s.t.}\quad&\Theta^Tx^{(i)}\ge1\quad\space\space\space \text{if}\quad y^{(i)}=1\\&\Theta^Tx^{(i)}\le-1\quad \text{if}\quad y^{(i)}=0\end{aligned}$

So our final objective is to calculate the following equation:

$\min_\theta\space \frac{1}{2}\sum_{j=1}^{n}\theta_j^2\\$

$\begin{aligned}\text{s.t.}\quad&\Theta^Tx^{(i)}\ge1\quad\space\space\space \text{if}\quad y^{(i)}=1\\&\Theta^Tx^{(i)}\le-1\quad \text{if}\quad y^{(i)}=0\end{aligned}$

**Note: SVM is sensitive to noise.**

<img src="pics\7-noise.png" style="zoom:50%;" />

### 7.2 - SVM Mathematical Inducement

Take $\Theta=\begin{bmatrix}\theta_1\\\theta_2\end{bmatrix}$ as an example: (regard $\theta_0$ as 0)

#### The Cost Function Part

$\begin{aligned}\min_\theta\space\frac{1}{2}\sum_{j=1}^{2}\theta_j^2&=\frac{1}{2}\big(\theta_1^2+\theta_2^2\big)\\&=\frac{1}{2}\bigg(\sqrt{\theta_1^2+\theta_2^2}\bigg)^2\\&=\frac{1}{2}\Theta^T\Theta\\&=\frac{1}{2}\big\|\Theta\big\|^2\end{aligned}$

#### The Condition Part

$\begin{aligned}\text{s.t.}\quad&\Theta^Tx^{(i)}\ge1\quad\space\space\space \text{if}\quad y^{(i)}=1\\&\Theta^Tx^{(i)}\le-1\quad \text{if}\quad y^{(i)}=0\end{aligned}$

Essentially $\Theta^Tx^{(i)}$ is the dot product of $\Theta$ and $x^{(i)}$ , it looks like:

<img src="pics\7-dot_product.png" style="zoom:50%;" /> <img src="pics\7-dot_product2.png" style="zoom:50%;" />

Let $p^{(i)}$ be the projection of $x^{(i)}$ to $\Theta$ , then we get:

$\begin{aligned}\text{s.t.}\quad&\big\|\Theta\big\|\cdot p^{(i)}\ge1\quad\space\space\space \text{if}\quad y^{(i)}=1\\&\big\|\Theta\big\|\cdot p^{(i)}\le-1\quad \text{if}\quad y^{(i)}=0\end{aligned}$

#### Summary

$\min_\theta\space\frac{1}{2}\big\|\Theta\big\|^2\\$

$\begin{aligned}\text{s.t.}\quad&\big\|\Theta\big\|\cdot p^{(i)}\ge1\quad\space\space\space \text{if}\quad y^{(i)}=1\\&\big\|\Theta\big\|\cdot p^{(i)}\le-1\quad \text{if}\quad y^{(i)}=0\end{aligned}$

To minimize $\frac{1}{2}\big\|\Theta\big\|^2\\$ , we need to maximize $\big|p^{(i)}\big|$ , which denotes the margin between the two classes.

<img src="pics\7-margin.png" style="zoom:50%;" />

### 7.3 - Kernels

For some non-linear classification problems, we can remap $x^{(i)}$ to a new feature $f^{(i)}$ using kernel function.

#### Procedure

**Given $x^{(1)},x^{(2)},\dots,x^{(m)}$ , we can apply kernel function using following steps:**

1. Choose each training example $x^{(i)}$ as the landmark $l^{(i)}$ .

   We have $l^{(i)}=x^{(i)}$ for all $i$ .

   Finally we acquire $m$ landmarks $l^{(1)},l^{(2)},\dots,l^{(m)}$ in total.

2. For each training example $x^{(i)}$ , using all $m$ landmarks to calculate new features $f^{(i)}$ .

   The new feature $f^{(i)}$ have $m$ dimensions, and $f_j^{(i)}=f(x^{(i)},l^{(j)})\\$ .

   Finally we acquire $m$ features $f^{(1)},f^{(2)},\dots,f^{(m)}$ in total.

3. Using $f^{(1)},f^{(2)},\dots,f^{(m)}$ to train the SVM model.

   The new training objective is $\min_\theta\space C\sum_{i=1}^{m}\big[y^{(i)}\text{cost}_1(\Theta^Tf^{(i)})+(1-y^{(i)})\text{cost}_0(\Theta^Tf^{(i)})\big]+\frac{1}{2}\sum_{j=1}^{n}\theta_j^2\\$ . ( $n=m$ )

$f^{(i)}$ : $m$ dimensions.

#### Kernel Selection

Linear Kernel: Same as no kernel.

Gaussian Kernel: $f(x,l)=\exp(-\frac{\|x-l\|^2}{2\sigma^2})\\$ .

Polynomial Kernel: $f(x,l)=(x^Tl+b)^a$ .

And so on...

#### Parameter Selection

|       | $\sigma^2$ (Gaussian Kernel)                                 | $C\ (=\frac{1}{\lambda})\\$ (Cost Function) |
| ----- | ------------------------------------------------------------ | ------------------------------------------- |
| Large | Features $f^{(i)}$ vary more smoothly. <br>**High bias.**    | Small $\lambda$ . <br>**High variance.**    |
| Small | Features $f^{(i)}$ vary less smoothly. <br/>**High variance.** | Large $\lambda$ . <br/>**High bias.**       |

#### SVM vs Logistic Regression vs Neural Network

Suppose we have $m$ training examples. Each example contains $n$ features.

|                                       | SVM                                         | Logistic Regression               | Neural Network                       |
| ------------------------------------- | ------------------------------------------- | --------------------------------- | ------------------------------------ |
| $n>m$ <br>(e.g. $n=100,m=10$ )        | **Linear Kernel.** <br>(avoid overfitting)  | Work fine.                        | Always work well.                    |
| $n<m$ <br>(e.g. $n=100,m=1000$ )      | **Gaussian Kernel.** <br>(or other kernels) | Work fine.<br>(SVM may be better) | Always work well.<br>(SVM is faster) |
| $n\ll m$ <br>(e.g. $n=100,m=100000$ ) | **Linear Kernel.**<br>(reduce time cost)    | Work fine.                        | Always work well.                    |



## 8 - Unsupervised Learning

### 8.1 - Clustering with K-means

#### Algorithm Steps

Assuming that we have $m$ training examples. We want to divide these examples into $K<m$ clusters.

1. Randomly pick $K$ training examples $x_1,x_2,\dots,x_K$ .

   Initialize $K$ cluster centroids $\mu_1,\mu_2,\dots,\mu_K$ using picked examples.

   We have $\mu_i=x_i$ .

2. Loop:

   For all $x^{(i)}$ , choose the nearest centroid as its cluster, denotes $c^{(i)}$ .

   For all clusters, update $\mu_k$ with the mean value of the points assigned to it.

3. Stop when there is no change in each cluster.

#### Optimization Objective

Minimize the average distance of all examples to their corresponding centroids.

$J(C,\Mu)=\frac{1}{m}\sum_{i=1}^m\|x^{(i)}-\mu_{c^{(i)}}\|\\$

#### Choosing the Number of Clusters $K$

Elbow Method:

Plot $J(C,\Mu)$ while increasing $K$ . Choose the "elbow point" as the answer,

More Importantly:

Evaluate based on a metric for how well it performs for later purpose.

### 8.2 - Principal Component Analysis (PCA)

#### Algorithm Steps

Assuming that we have $m$ training examples. Each example is a $n$ dimensional vector.

We want to compose each example into a $k$ dimensional vector. ( $k<n$ ) 

1. Do feature scaling or mean normalization.

2. Compute "covariance matrix" $\Sigma$ :

   $\Sigma=\frac{1}{m}\sum_{i=1}^nx^{(i)}\big(x^{(i)}\big)^T\\$ ( $n\times n$ )

3. Compute "eigenvector matrix" $U$ and "eigenvalue matrix" $V$ of $\Sigma$ :

   $U=\begin{bmatrix}u^{(1)}&u^{(2)}&\cdots&u^{(n)}\end{bmatrix}$ ( $n\times n$ )

   $V=\begin{bmatrix}v^{(1)}&v^{(2)}&\cdots&v^{(n)}\end{bmatrix}$ ( $1\times n$ )

   $u^{(i)}$ is the $i$ th eigenvector of $\Sigma$ . ( $n\times 1$ )

   $v^{(i)}$ is the $i$ th eigenvalue of $\Sigma$ .

4. Select the largest $k$ eigenvalues from $V$ , concatenate the corresponding $k$ eigenvectors together as a new matrix $U'$ .

   $U'=\begin{bmatrix}u'^{(1)}&u'^{(2)}&\cdots&u'^{(k)}\end{bmatrix}$ ( $n\times k$ )

5. Compute new features matrix $Z$ .

   $Z=XU'=\begin{bmatrix}z^{(1)}\\z^{(2)}\\\vdots\\z^{(n)}\end{bmatrix}$ ( $m\times k$ )

#### Optimization Objective

Minimize the average distance of all examples to the hyperplane.

$\min_U\space\frac{1}{m}\sum_{i=1}^md\big(x^{(i)},U\big)\\$

#### Choosing the Number of Principal Components $K$

We can reconstruct approximately our $n$ dimensional examples back with some error:

$X_{re}=Z(U')^T=\begin{bmatrix}x_{re}^{(1)}\\x_{re}^{(2)}\\\vdots\\x_{re}^{(n)}\end{bmatrix}$ ( $m\times n$ )

We can then compute the average information loss:

$L=\frac{1}{m}\sum_{i=1}^m\|x^{(i)}-x_{re}^{(i)}\|^2\\$

So the loss rate is:

$r=\frac{\frac{1}{m}\sum_{i=1}^m\|x^{(i)}-x_{re}^{(i)}\|^2}{\frac{1}{m}\sum_{i=1}^m\|x^{(i)}\|^2\\}=\frac{\sum_{i=1}^m\|x^{(i)}-x_{re}^{(i)}\|^2}{\sum_{i=1}^m\|x^{(i)}\|^2\\}\\$

We can choose the smallest value of $k$ that satisfies:

$r<0.01$ (99% of variance retained)

#### Optimization Objective (recap)

With the reconstructed examples, our training objective can also be described as:

$\min_U\space\frac{1}{m}\sum_{i=1}^m\|x^{(i)}-x_{re}^{(i)}\|^2\\$

#### Applications

1. Speed up computation by composing features.
2. Vasualization.

**Notice:** It is a bad way to use PCA to preventing overfitting. ( This might work, but why not use regularization instead? )



## 9 - Anomaly Detection

### 9.1 - Density Estimation

#### Problem Motivation

We have a **very skewed dataset** $\big\{x^{(1)},x^{(2)},\dots,x^{(m)}\big\}$ , in which the number of negative examples is much larger than that of positive ones (e.g. $m_0=10000,m_1=20$ ). Our objective is to detect the **anomaly examples** (positive ones).

One possible way is to use **supervised learning algorithms** to build a classification model. But we have too little positive examples that our model can't fit all possible "types" of anomaly examples. So future anomaly examples may looking nothing like the previous ones we used for training. **As a result, our model using supervised learning algorithm may behave quite bad.**

To handle extreme datasets like this, we need to use another method called **"density estimation"**.

#### Algorithm Formulation

Suppose we have $m$ training examples (all negative). Each example contains $n$ features.

1. Assume that our training examples follow a specific distribution $D\big(\Theta\big)$ , we have:

   $x^{(i)}\sim D\big(\Theta\big)$ for $i=1,2,\dots,m$

2. We can then estimate parameters $\Theta$ and fit this distribution.

3. For a new example $x^{new}$ , we can calculate the possibility that $x^{new}$ follows the distribution $D$ :

   $p^{new}=P\big(x_1^{new},x_2^{new},\dots,x_n^{new};\Theta\big)$

4. If $p^{new}<\epsilon$ , then we think that $x^{new}$ is a anomaly point.

#### Distribution Selection

**Single Variate Gaussian Distribution:**

Each feature follows a single variate gaussian distribution. (all features are independent of each other)

$x_k\sim N\big(\mu_k,\sigma_k^2\big)$ for $k=1,2,\dots,n$

$p^{new}=\prod_{k=1}^n P_k\big(x_k^{new};\mu_k,\sigma_k^2\big)\\$

**Multivariate Gaussian Distribution:**

All features together follow a $n$ variate gaussian distribution.

$x_1,x_2,\dots,x_n\sim N\big(\mu_1,\mu_2,\dots,\mu_n,\sigma_1^2,\sigma_2^2,\dots,\sigma_n^2\big)$

$p^{new}=P\big(x_1^{new},x_2^{new},\dots,x_n^{new};\mu_1,\mu_2,\dots,\mu_n,\sigma_1^2,\sigma_2^2,\dots,\sigma_n^2)\\$

#### Dataset Division

|                   | Training Set | Cross Validation Set | Test Set |
| ----------------- | ------------ | -------------------- | -------- |
| Negative Examples | 60%          | 20%                  | 20%      |
| Positive Examples | 0%           | 50%                  | 50%      |

### 9.2 - Recommender System

#### Content Based

<img src="pics\9-movie.png" style="zoom:50%;" />

**Suppose we have $n_m$ movies and $n_u$ users.**

$r(i,j)$ : equals 1 if user $j$ has rated movie $i$ (else 0)

$m^{(j)}$ : number of movies rated by user $j$

$m'^{(i)}$ : number of users rated movie $i$

$x^{(i)}$ : feature vector of movie $i$

$\theta^{(j)}$ : parameter vector of user $j$

$y^{(i,j)}$ : rating by user $j$ on movie $i$

**We can train a linear regression model for every user.**

For movie $i$, user $j$ , our predicted rating is ${\theta^{(j)}}^Tx^{(i)}$ .

The cost function for user $j$ is:

$J\big(\theta^{(j)}\big)=\frac{1}{2m^{(j)}}\sum_{i:r(i,j)=1}^{m^{(j)}}\Big({\theta^{(j)}}^Tx^{(i)}-y^{(i,j)}\Big)^2+\frac{\lambda}{2m^{(j)}}\sum_{k=1}^n\big(\theta_k^{(j)}\big)^2\\$

Combine all $J\big(\theta^{(j)}\big)$ together, the global cost function is:

$J\big(\theta^{(1)},\theta^{(2)},\dots,\theta^{(n_u)}\big)=\frac{1}{2n_m}\sum_{j=1}^{n_u}\sum_{i:r(i,j)=1}^{m^{(j)}}\Big({\theta^{(j)}}^Tx^{(i)}-y^{(i,j)}\Big)^2+\frac{\lambda}{2n_m}\sum_{j=1}^{n_u}\sum_{k=1}^n\big(\theta_k^{(j)}\big)^2\\$

#### Collaborative Filtering

Given $x^{(1)},x^{(2)},\dots,x^{(n_m)}$ , we can estimate $\theta^{(1)},\theta^{(2)},\dots,\theta^{(n_u)}$ by minimizing:

$J\big(\theta^{(1)},\theta^{(2)},\dots,\theta^{(n_u)}\big)=\frac{1}{2n_m}\sum_{j=1}^{n_u}\sum_{i:r(i,j)=1}^{m^{(j)}}\Big({\theta^{(j)}}^Tx^{(i)}-y^{(i,j)}\Big)^2+\frac{\lambda}{2n_m}\sum_{j=1}^{n_u}\sum_{k=1}^n\big(\theta_k^{(j)}\big)^2\\$

Similarly, given $\theta^{(1)},\theta^{(2)},\dots,\theta^{(n_u)}$ , we can also estimate $x^{(1)},x^{(2)},\dots,x^{(n_m)}$ by minimizing:

$J\big(x^{(1)},x^{(2)},\dots,x^{(n_m)}\big)=\frac{1}{2n_u}\sum_{i=1}^{n_m}\sum_{j:r(i,j)=1}^{m'^{(i)}}\Big({\theta^{(j)}}^Tx^{(i)}-y^{(i,j)}\Big)^2+\frac{\lambda}{2n_u}\sum_{i=1}^{n_m}\sum_{k=1}^n\big(x_k^{(i)}\big)^2\\$

Notice that both function have the same objective with the regularization term removed:

$\min\space\frac{1}{2}\sum_{(i,j):r(i,j)=1}^{(n_m,n_u)}\Big({\theta^{(j)}}^Tx^{(i)}-y^{(i,j)}\Big)^2\\$

So we can combine these two cost functions together:

$J\big(x^{(1)},\dots,x^{(n_m)},\theta^{(1)},\dots,\theta^{(n_u)}\big)=\frac{1}{2}\sum_{(i,j):r(i,j)=1}^{(n_m,n_u)}\Big({\theta^{(j)}}^Tx^{(i)}-y^{(i,j)}\Big)^2+\frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^n\big(x_k^{(i)}\big)^2+\frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^n\big(\theta_k^{(j)}\big)^2\\$

We can randomly initialize $x^{(1)},\dots,x^{(n_m)},\theta^{(1)},\dots,\theta^{(n_u)}$ and use gradient descent algorithm to estimate the parameters.

#### Mean Normalization

<img src="pics\9-movie2.png" style="zoom:50%;" />

For a person who hasn't rated any movie, the objective turns out to be:

$\begin{aligned}J\big(\theta^{(j)}\big)&=\frac{1}{2m^{(j)}}\sum_{i:r(i,j)=1}^{m^{(j)}}\Big({\theta^{(j)}}^Tx^{(i)}-y^{(i,j)}\Big)^2+\frac{\lambda}{2m^{(j)}}\sum_{k=1}^n\big(\theta_k^{(j)}\big)^2\\&=\frac{\lambda}{2m^{(j)}}\sum_{k=1}^n\big(\theta_k^{(j)}\big)^2\end{aligned}$

So the estimates parameter will be like $\theta_1^{(j)}=\theta_2^{(j)}=\dots=\theta_n^{(j)}=0$ .

It is unreasonable to predict that a person will rate every movie 0 score.

**We can fix this problem by applying mean normalization for each movie:**

$\overline{y^{(i)}}=\frac{1}{m'^{(i)}}\sum_{j:r(i,j)=1}^{n_u}y^{(i,j)}\\$ (mean rating of movie $i$)

$y^{(i,j)}:=y^{(i,j)}-\overline{y^{(i)}}\\$

<img src="pics\9-movie3.png" style="zoom:50%;" />

Then our predicted rating "0" will be a neutral score.



## 10 - Large Scale Machine Learning

### 10.1 - Stochastic Gradient Descent

| (Each Iteration)   | (Batch) Gradient Descent                                     | Stochastic Gradient Descent                                | Mini-batch Gradient Descent                                  |
| ------------------ | ------------------------------------------------------------ | ---------------------------------------------------------- | ------------------------------------------------------------ |
| number of examples | $m$                                                          | $1$                                                        | $b<m$                                                        |
| cost function      | $J(\theta)=\frac{1}{m}\sum_{i=1}^m\text{Cost}\big(h_\theta(x^{(i)}),y^{(i)}\big)$ | $J(\theta)=\text{Cost}\big(h_\theta(x^{(i)}),y^{(i)}\big)$ | $J(\theta)=\frac{1}{b}\sum_{i=1}^b\text{Cost}\big(h_\theta(x^{(i)}),y^{(i)}\big)$ |

| (Each Epoch)         | (Batch) Gradient Descent | Stochastic Gradient Descent | Mini-batch Gradient Descent |
| -------------------- | ------------------------ | --------------------------- | --------------------------- |
| number of examples   | $m$                      | $m'<m$                      | $b<m$                       |
| number of iterations | $1$                      | $m'$                        | $1$                         |
| randomly shuffle     | No                       | Yes                         | Yes                         |

Choosing $m'$ : An example is $m'=0.1m$

### 10.2 - Online Learning

For some online learning problem, we have a stream of data that changes every time.

We can then train our model **only based on the latest data** and **discard the used data**.

Then our model can **fit the change of data stream** and get the latest features.

### 10.3 - Map Reduce

<img src="pics\10-map_reduce.png" style="zoom:50%;" />



## 11 - Problem Description and Pipeline

### Real Life Example: Photo OCR

#### **Sliding Window Algorithm**

1. Repeatedly move a window in a fixed step to detect different parts of a picture.

2. If the content in a window part is likely to be "text", mark it as "text parts".
3. For all continuous "text parts", use a smaller window to do character segmentation.
4. For all characters, do character recognition.

#### Getting More Data

**Artificial Data:**

We can generate data autonomously using font libraries in the computer.

In this way, we have theoretically unlimited training data.

**Crowd Source:**

Hiring the crowd to label data.

**Special Attentions:**

1. Make sure we have a low-bias classifier, otherwise we should increase features in our model instead.

2. How much time it will save us if using artificial data rather than collecting real life data.





