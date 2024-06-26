# Deep Learning in Applications Final Test

## Name: Brayan Duran Medina

### 1. RL problem statement. State, Action, Reward, Environment, Agent

Reinforcement Learning (RL) is an area of machine learning concerned with how software agents should take actions in an environment in order to maximize some notion of cumulative reward. RL is modeled as a decision-making process where an agent learns to achieve a goal in an uncertain, potentially complex environment. Here's an explanation of the key components of an RL problem statement:

**State:** Represents the agent's current situation within the environment, encapsulating all necessary information for decision-making. For instance, in chess, it's the positions of all pieces.

**Action:** The decision an agent makes at each step, determined by the available set of actions (action space) from a given state. For example, in autonomous driving, actions might include accelerating or steering.

**Reward:** Immediate feedback given to an agent after an action, used to evaluate the action's effectiveness. The goal is to maximize cumulative rewards, balancing short and long-term benefits.

**Environment:** The external world in which the agent operates, providing states and responding to the agent’s actions with new states and rewards.

**Agent:** The entity that interacts with the environment, making decisions based on states, executing actions, and learning from the resulting rewards to optimize its policy for maximum cumulative rewards.

### 2. Crossentropy method

The Cross-Entropy Method (CEM) in Reinforcement Learning (RL) is a straightforward and efficient optimization technique that is especially useful for solving high-dimensional, non-convex optimization problems such as those found in many RL environments. Unlike many other RL algorithms that attempt to learn the value of actions directly or model the environment, CEM is a model-free policy search method. It focuses on finding a policy that maximizes the expected return by sampling and updating policies based on their performance. The steps involved in the Cross-Entropy Method for RL typically include:

**Initialization:**
Begin with an initial policy, often defined by a probability distribution over actions. In many cases, this can be represented by parameters of a probability distribution, such as the mean and variance if actions are continuous.

**Sampling:**
Generate a batch of episodes by simulating the current policy. Each episode consists of states, actions taken, and rewards received from the environment.

**Evaluation:**
At the end of each episode, compute the total reward (or return). This step involves simply summing up the rewards obtained in each episode.

**Selection:**
Sort the episodes based on their total rewards and select the top-performing episodes. The selection criterion typically involves choosing a percentile (e.g., the top 10% or 20%). This subset contains the "elite" samples that yield higher returns.

**Update:**
Update the parameters of the policy's probability distribution based on the actions in the elite episodes. For example, if actions are modeled using a Gaussian distribution, update the mean and variance towards those actions that led to higher returns.

**Iteration:**
Repeat the process of sampling, evaluation, selection, and updating for a set number of iterations or until convergence criteria are met, such as minimal changes in the policy parameters or achieving a satisfactory level of performance.

### 3. Value function, Q-function

#### Value Function

The value function, often denoted as $V(s)$, measures the expected cumulative reward an agent can achieve starting from a specific state $s$ and following a particular policy $\pi$. The policy $\pi$ is a strategy that the agent employs, which dictates the actions to take in different states. The value function thus answers the question, "What is the long-term reward that the agent can expect by being in a certain state and following a specific policy?"

Mathematically, it's defined as:
$V^\pi(s) = \mathbb{E}[R_t | s_t = s, \pi]$
where $R_t$ is the total reward starting from time $t$, conditioned on the state being $s$ and following policy $\pi$. The expectation $\mathbb{E}$ accounts for the probabilistic nature of the state transitions and the rewards.

#### Q-Function (Action-Value Function)

The Q-function, or action-value function, extends the value function by incorporating actions. Denoted as $Q(s, a)$, it measures the expected cumulative reward starting from state $s$, taking an action $a$, and thereafter following policy $\pi$. It effectively evaluates the usefulness of taking a particular action in a given state.

The Q-function is formally defined as:
$Q^\pi(s, a) = \mathbb{E}[R_t | s_t = s, a_t = a, \pi]$
This function gives insight into which actions are better in a particular state by providing a value associated with each action, thus guiding the decision-making process.

### 4. Q-learning, approximate Q-learning. DQN, bells and whistles (Experience replay, Double DQN, autocorrelation problem)

#### Q-learning

Q-learning is a model-free reinforcement learning algorithm that seeks to learn the optimal action-value function, $Q^\*(s, a)$, which provides the best action in every state. The core idea is to estimate the Q-values based on the Bellman equation iteratively. The Q-values are updated as follows:

$Q(s, a) \leftarrow Q(s, a) + \alpha \left[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right]$
where $s$ is the current state, $a$ is the action taken, $r$ is the reward received, $s'$ is the new state, $\alpha$ is the learning rate, and $\gamma$ is the discount factor. The term $\max\_{a'} Q(s', a')$ represents the highest Q-value achievable from the next state $s'$, providing a lookahead that guides the updates.

#### Approximate Q-learning

As the state or action spaces become very large, storing and updating a Q-value for every state-action pair becomes computationally infeasible. Approximate Q-learning addresses this by using function approximation to estimate the Q-values. Instead of maintaining a table, a parameterized function, typically a linear function or a neural network, is used to approximate $Q(s, a; \theta)$, where $\theta$ are the parameters of the function. The parameters are adjusted to minimize the difference between the estimated Q-values and the target Q-values obtained from the Bellman equation.

#### Deep Q-Network (DQN)

Deep Q-Networks (DQN) extend approximate Q-learning by using deep neural networks to approximate the Q-function. Introduced by researchers at DeepMind, DQN was a breakthrough because it demonstrated that a neural network could effectively learn to play Atari games at a superhuman level directly from pixel inputs. DQNs stabilize the training of deep neural networks with two key innovations:

-   **Experience Replay:** This technique stores the agent's experiences at each time step, $(s, a, r, s')$, in a data set called a replay buffer. Random mini-batches from this buffer are used to update the network, which helps in breaking the correlation between consecutive samples, thus stabilizing the learning process.

-   **Fixed Q-targets:** DQN uses a separate, fixed network to generate the target Q-values in the update equation. The weights of this target network are updated less frequently (every few thousand steps) than the primary network weights. This approach reduces the correlations with the target, further stabilizing learning.

#### Enhancements to DQN

-   **Double DQN:** In standard DQN, the same values are used both to select and to evaluate an action, which can lead to overoptimistic value estimates. Double DQN addresses this by decoupling the selection from the evaluation. Two networks are used: one to select the best action and another to evaluate the action. This reduces overestimations and leads to more stable and reliable learning.

-   **Addressing Autocorrelation Problem:** The autocorrelation problem in RL refers to the issue where consecutive states are often highly correlated, leading to biased samples and inefficient learning. Techniques like experience replay help mitigate this problem by mixing more distant experiences in each batch, thus reducing the immediate correlations.

### 5. Policy gradient and REINFORCE algorithm

-   a. Baseline idea

#### Policy Gradient

Policy gradient methods are a class of algorithms in reinforcement learning that optimize the policy directly. A "policy" in RL is a strategy used by the agent to decide which action to take given a state. In policy gradient methods, the objective is to adjust the policy parameters $\theta$ in a way that maximizes the expected return, which is the cumulative reward the agent receives over time.

The core idea is that you can use gradient ascent to find the best parameters that increase the expected return. The policy is often modeled as a probability distribution $\pi(a|s, \theta)$ that defines the likelihood of taking action $a$ in state $s$ with parameters $\theta$. The gradient of the expected return with respect to the policy parameters $\theta$ guides how the parameters should be adjusted. The update rule typically looks like this:

$\theta*{\text{new}} = \theta*{\text{old}} + \alpha \nabla\_\theta J(\theta)$

where $\alpha$ is the learning rate and $\nabla\_\theta J(\theta)$ is the gradient of the performance measure $J(\theta)$ with respect to $\theta$.

#### REINFORCE Algorithm

REINFORCE is a specific type of policy gradient method. It is also known as Monte Carlo Policy Gradient, as it relies on complete episodes for training, using their returns for the policy gradient estimate. The algorithm works as follows:

1. **Generate an episode:** Using the current policy $\pi\_\theta$, run the agent through an episode and record all states, actions, and rewards.
2. **Calculate returns:** For each time step $t$ in the episode, calculate the total discounted return $G_t$ from that time step onward.
3. **Policy update:** Use the returns as an unbiased sample of the policy gradient. The policy parameters are updated by:

    $\theta = \theta + \alpha \sum*{t=0}^{T-1} \nabla*\theta \log \pi\_\theta(a_t|s_t) G_t$

    Here, $\nabla*\theta \log \pi*\theta(a_t|s_t)$ is the gradient of the log-probability of the action taken, and $G_t$ is the return from time step $t$.

#### Baseline Idea

A major enhancement in policy gradient methods, including REINFORCE, is the use of a "baseline." A baseline is typically a function that estimates the expected reward from a given state. By subtracting this baseline from the return, the variance of the gradient estimate can be reduced, leading to more stable and efficient learning. The baseline does not bias the gradient estimates if chosen appropriately.

One common choice for the baseline is a state-value function $V(s)$, which estimates how good it is to be in a particular state. The REINFORCE update rule with a baseline $b(s_t)$ looks like this:

$\theta = \theta + \alpha \sum*{t=0}^{T-1} \nabla*\theta \log \pi\_\theta(a_t|s_t) (G_t - b(s_t))$

This modification helps in reducing the variance of the update step, making learning smoother and often faster.

### 6. Policy gradient applications in other domains (outside RL). How Self-Critical Sequence Training is performed? What is used as a baseline?

#### Policy Gradient Applications Beyond RL

Policy gradient methods are used in domains requiring sequential decision-making or discrete outcome optimization, such as:

1. **Natural Language Processing (NLP):** For tasks like machine translation or dialogue generation, where sequences of words are generated similarly to actions in RL.
2. **Robotics:** In imitation learning, where the aim is to mimic expert behavior rather than maximizing a reward.
3. **Optimization:** In solving complex combinatorial problems by framing them as a series of decision steps.

#### Self-Critical Sequence Training (SCST)

SCST is a method applied in image captioning, adapting policy gradient techniques to maximize a reward function related to caption quality.

**Steps in SCST:**

1. **Generate Captions:** Produce captions by sampling from the model and by greedy selection.
2. **Compute Reward:** Use metrics like BLEU or CIDEr to evaluate both sets of captions against reference captions.
3. **Update Parameters:** Use the difference in reward between sampled and greedy captions as an advantage to update the model. This encourages the model to generate captions better than or equal to the greedy captions.

#### Baseline in SCST

The baseline in SCST is the reward of the greedy captions. This baseline helps stabilize training by normalizing rewards of the sampled captions, pushing the model to generate better outputs.
