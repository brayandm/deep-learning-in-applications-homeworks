# Deep Learning in Applications Final Test

## Name: Brayan Duran Medina

### 1. RL problem statement. State, Action, Reward, Environment, Agent

Given:

-   Objects $x \in \mathcal{X}$
-   Loss function $L(\hat{y}, y)$
-   Model Family $f \in \mathcal{F}$, $f:X \rightarrow Y$

Goal:

-   Find optimal mapping $f^* = \arg \min_{f } L(f(x), y)$

Reinforcement Learning (RL) is an area of machine learning concerned with how software agents should take actions in an environment in order to maximize some notion of cumulative reward. RL is modeled as a decision-making process where an agent learns to achieve a goal in an uncertain, potentially complex environment. Here's an explanation of the key components of an RL problem statement:

**State:** Represents the agent's current situation within the environment, encapsulating all necessary information for decision-making. For instance, in chess, it's the positions of all pieces.

**Action:** The decision an agent makes at each step, determined by the available set of actions (action space) from a given state. For example, in autonomous driving, actions might include accelerating or steering.

**Reward:** Immediate feedback given to an agent after an action, used to evaluate the action's effectiveness. The goal is to maximize cumulative rewards, balancing short and long-term benefits.

**Environment:** The external world in which the agent operates, providing states and responding to the agent’s actions with new states and rewards.

**Agent:** The entity that interacts with the environment, making decisions based on states, executing actions, and learning from the resulting rewards to optimize its policy for maximum cumulative rewards.

### 2. Crossentropy method

### 3. Value function, Q-function

### 4. Q-learning, approximate Q-learning. DQN, bells and whistles (Experience replay, Double DQN, autocorrelation problem)

### 5. Policy gradient and REINFORCE algorithm

-   a. Baseline idea

### 6. Policy gradient applications in other domains (outside RL). How Self-Critical Sequence Training is performed? What is used as a baseline?
