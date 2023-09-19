# POLICY EVALUATION

## AIM:
To develop a Python program to evaluate the given policy.
## PROBLEM STATEMENT:
The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state. The environment is slippery, so the agent has a chance of moving in the opposite direction of the action it takes.

### States
### The environment has 7 states:
Two Terminal States: G: The goal state & H: A hole state.
Five Transition states / Non-terminal States including S: The starting state.
Actions
The agent can take two actions:

R: Move right.
L: Move left.
### Transition Probabilities
The transition probabilities for each action are as follows:

50% chance that the agent moves in the intended direction.
33.33% chance that the agent stays in its current state.
16.66% chance that the agent moves in the opposite direction.
For example, if the agent is in state S and takes the "R" action, then there is a 50% chance that it will move to state 4, a 33.33% chance that it will stay in state S, and a 16.66% chance that it will move to state 2.

### Rewards:
The agent receives a reward of +1 for reaching the goal state (G). The agent receives a reward of 0 for all other states.

### Graphical Representation
![image](https://github.com/gpavithra673/rl-policy-evaluation/assets/93427264/d77c8d45-e3d2-46cb-b17b-37dbaf294636)

## POLICY EVALUATION FUNCTION
Formula
![image](https://github.com/gpavithra673/rl-policy-evaluation/assets/93427264/ac0a94b9-3cf8-42c0-a8ea-33a01d6192af)
## PROGRAM:
```
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P), dtype=np.float64)
    # Write your code here to evaluate the given policy
    while True:
      V = np.zeros(len(P))
      for s in range(len(P)):
        for prob, next_state, reward, done in P[s][pi(s)]:
          V[s] += prob * (reward + gamma *  prev_V[next_state] * (not done))
      if np.max(np.abs(prev_V - V)) < theta:
        break
      prev_V = V.copy()
    return V

## Code to evaluate the first policy
V1 = policy_evaluation(pi_1, P)
print_state_value_function(V1, P, n_cols=7, prec=5)

## Code to evaluate the second policy
V2 = policy_evaluation(pi_2, P)
print_state_value_function(V2, P, n_cols=7, prec=5)

## Comparing policies based on state value function
### The state value function of the second policy V2 is greater than that of the first policy V1, so we conclude that the second policy is the best policy.

V1
print_state_value_function(V1, P, n_cols=7, prec=5)
V2
print_state_value_function(V2, P, n_cols=7, prec=5)
V1>=V2
if(np.sum(V1>=V2)==7):
  print("The first policy is the better policy")
elif(np.sum(V2>=V1)==7):
  print("The second policy is the better policy")
else:
  print("Both policies have their merits.")
```
## OUTPUT:
### POLICY 1:
![image](https://github.com/gpavithra673/rl-policy-evaluation/assets/93427264/388c1557-8e6d-49a1-8a6e-1743ad60b225)
![image](https://github.com/gpavithra673/rl-policy-evaluation/assets/93427264/2227af81-dfaf-4554-9a41-8422d83c2de1)
![image](https://github.com/gpavithra673/rl-policy-evaluation/assets/93427264/f1a0dc14-1627-453a-a1d8-1ce0c74727b0)

### POLICY 2:
![image](https://github.com/gpavithra673/rl-policy-evaluation/assets/93427264/1f01dc55-0361-4686-9c26-9132d3285f0d)
![image](https://github.com/gpavithra673/rl-policy-evaluation/assets/93427264/162e1abb-6d7d-4ca4-a5bc-4e5657f1aa79)
![image](https://github.com/gpavithra673/rl-policy-evaluation/assets/93427264/e039060b-0e8a-4858-a0c6-49ac545819c2)

### COMPARISON:
![image](https://github.com/gpavithra673/rl-policy-evaluation/assets/93427264/8dd67c2c-7fab-4c9a-b81e-cecf18f7c6c9)

### CONCLUSION:
![image](https://github.com/gpavithra673/rl-policy-evaluation/assets/93427264/85b29801-71e6-4bf3-8c39-a55b33a1fd34)

## RESULT:
### Thus, a Python program is developed to evaluate the given policy.

Write your result here
