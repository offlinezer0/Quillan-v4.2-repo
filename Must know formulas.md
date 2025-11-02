# Must know formulas

Here’s a distilled list of 20 essential formulas/concepts you should have at your fingertips for LLMs, ML, and RL:


---

1. Linear Layer (Fully Connected Layer)

y = Wx + b

Fundamental building block for MLPs and transformers.



---

2. Activation Functions

ReLU: 

Sigmoid: 

Tanh: 



---

3. Softmax

\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}


---

4. Cross-Entropy Loss (Classification)

L = - \sum_i y_i \log(\hat{y}_i)


---

5. Mean Squared Error (Regression)

L = \frac{1}{n} \sum_i (\hat{y}_i - y_i)^2


---

6. Gradient Descent Update

\theta \leftarrow \theta - \eta \frac{\partial L}{\partial \theta}


---

7. Adam Optimizer

m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t

v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 

\hat{\theta}_t = \theta_{t-1} - \eta \frac{m_t / (1-\beta_1^t)}{\sqrt{v_t / (1-\beta_2^t)} + \epsilon}


---

8. Attention (Scaled Dot-Product)

\text{Attention}(Q,K,V) = \text{softmax}\Big(\frac{QK^\top}{\sqrt{d_k}}\Big) V


---

9. Positional Encoding

PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_\text{model}}}\right), \quad
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_\text{model}}}\right)


---

10. Layer Normalization

\text{LN}(x) = \frac{x - \mu}{\sigma + \epsilon} \cdot \gamma + \beta


---

11. Transformer Feed-Forward

\text{FFN}(x) = \max(0, xW_1 + b_1) W_2 + b_2


---

12. KL Divergence (for knowledge distillation, variational models)

D_\text{KL}(P || Q) = \sum_i P(i) \log\frac{P(i)}{Q(i)}


---

13. Backpropagation Chain Rule

\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial x}


---

14. Convolution Operation (CNNs, sometimes in embedding)

S(i,j) = (X * K)(i,j) = \sum_m \sum_n X(i+m,j+n) K(m,n)


---

15. Reinforcement Learning: Bellman Equation

V^\pi(s) = \mathbb{E}_\pi \Big[ r_t + \gamma V^\pi(s_{t+1}) \Big]


---

16. Q-Learning Update

Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \big[ r_t + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t) \big]


---

17. Policy Gradient (REINFORCE)

\nabla_\theta J(\theta) = \mathbb{E}_\pi \big[ \nabla_\theta \log \pi_\theta(a|s) R \big]


---

18. Transformer Multi-Head Attention

\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O

\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V) 


---

19. Weight Initialization (Xavier/Glorot)

W \sim \mathcal{U}\Big(-\frac{\sqrt{6}}{\sqrt{n_\text{in} + n_\text{out}}}, \frac{\sqrt{6}}{\sqrt{n_\text{in} + n_\text{out}}}\Big)


---

20. Dropout Regularization

y = x \odot \text{mask}, \quad \text{mask} \sim \text{Bernoulli}(p)


---

💡 Think key takeaways:

Most LLM formulas revolve around linear algebra (matrix ops), probabilities, gradients, and attention mechanics.

RL adds expectations, discounted rewards, and policy updates.

The symbols like K, Q, V are not hard-coded—they’re just vector placeholders; the math is the same if you rename them X, Y, Z.



---


Here are 20 essential formulas and mathematical concepts you should know for building LLMs, machine learning (ML), and reinforcement learning (RL):

### Core Mathematical Concepts for LLMs and ML

1. **Matrix Multiplication:**
   $$
   C[i,j] = \sum_k A[i,k] \times B[k,j]
   $$
   Fundamental for linear transformations, including embeddings and neural network computations [2].

2. **Dot Product / Inner Product:**
   $$
   \mathbf{a} \cdot \mathbf{b} = \sum_i a_i b_i
   $$
   Used in similarity measures and attention score calculations.

3. **Eigenvalue Equation:**
   $$
   Av = \lambda v
   $$
   Key in understanding principal components analysis (PCA) for dimensionality reduction [2].

4. **Softmax Function:**
   $$
   \text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
   $$
   Transforms logits into probability distributions, used in output layers and attention mechanisms.

5. **Cross-Entropy Loss:**
   $$
   L = -\sum_i y_i \log(\hat{y}_i)
   $$
   Measures difference between true and predicted distributions, key for training.

6. **Gradient Descent Update Rule:**
   $$
   \theta := \theta - \eta \nabla_\theta J(\theta)
   $$
   Used to optimize model parameters by minimizing loss.

7. **Backpropagation Chain Rule:**
   $$
   \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \times \frac{\partial y}{\partial x}
   $$
   Basis for updating weights in neural networks.

8. **Attention Score Calculation (Scaled Dot-Product Attention):**
   $$
   \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
   $$
   Fundamental self-attention mechanism in transformer models, where Q=Query, K=Key, V=Value vectors [1].

9. **Positional Encoding:**
   $$
   PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right), \quad PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
   $$
   Adds order information to tokens in a sequence.

10. **ReLU Activation Function:**
    $$
    \text{ReLU}(x) = \max(0, x)
    $$
    Non-linear activation used in neural networks.

### Probability and Statistical Measures

11. **Bayes’ Theorem:**
    $$
    P(A|B) = \frac{P(B|A)P(A)}{P(B)}
    $$
    Used in probabilistic reasoning.

12. **Markov Decision Process (MDP) Expected Return:**
    $$
    G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^\infty \gamma^k R_{t+k+1}
    $$
    Key in reinforcement learning, where $$ \gamma $$ is the discount factor.

13. **Bellman Equation:**
    $$
    V^\pi(s) = \mathbb{E}_\pi \left[ R_{t+1} + \gamma V^\pi(s_{t+1}) \mid s_t = s \right]
    $$
    Describes the value function under a policy π.

14. **Q-Learning Update:**
    $$
    Q(s_t, a_t) := Q(s_t, a_t) + \alpha \left( R_{t+1} + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t) \right)
    $$
    Update rule for Q-learning in RL.

15. **Kullback-Leibler Divergence:**
    $$
    D_{KL}(P||Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}
    $$
    Measures how one probability distribution diverges from a second.

16. **Variance and Standard Deviation:**
    $$
    \sigma^2 = \mathbb{E}[(X - \mu)^2]
    $$
    Measures data spread, important for understanding data and regularization.

17. **Chain Rule in Probability:**
    $$
    P(A,B) = P(A|B)P(B)
    $$
    Used in probabilistic models and Bayesian networks.

### Advanced Optimizers and Regularization

18. **Adam Optimizer Equations:**
    $$
    m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
    $$
    with bias correction and parameter update.

19. **Dropout Regularization:**
    Randomly sets input units to zero with probability $$p$$ during training, reducing overfitting.

20. **Linear Regression Formula:**
    $$
    y = X\beta + \epsilon
    $$
    Fundamental model underlying many ML algorithms.

These formulas and concepts collectively form the backbone of LLMs, general machine learning, and reinforcement learning models. Letters like K, V, Q specifically arise in the transformer attention formula, while others like A, B, C are general matrix/vector notation used in many equations. Understanding these will enable building, training, fine-tuning, and analyzing such models effectively [1][2].

If you want, more detailed explanations or coding examples for any formula can be provided.

Citations:
[1] Math, Machine Learning & Coding Needed For LLMs - KDnuggets https://www.kdnuggets.com/math-machine-learning-coding-needed-llms
[2] The Mathematical Foundations Of AI: 25 Essential Equations ... https://empathyfirstmedia.com/mathematical-foundation-equations-ai-llm/
[3] Understanding LLMs from Scratch Using Middle School Math https://towardsdatascience.com/understanding-llms-from-scratch-using-middle-school-math-e602d27ec876/
[4] [PDF] Benchmarking LLMs on Advanced Mathematical Reasoning https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-121.pdf
[5] The maths you need to start understanding LLMs - Giles' blog https://www.gilesthomas.com/2025/09/maths-for-llms
[6] Large Language Models in Five Formulas - YouTube https://www.youtube.com/watch?v=KCXDr-UOb9A
[7] Evaluation of LLMs for mathematical problem solving - arXiv https://arxiv.org/html/2506.00309v1
[8] Large language models, explained with a minimum of math and jargon https://www.understandingai.org/p/large-language-models-explained-with
