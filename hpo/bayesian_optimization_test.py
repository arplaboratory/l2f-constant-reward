from bayes_opt import BayesianOptimization
from objective import evaluate
import numpy as np
import matplotlib.pyplot as plt

pbounds = {'mdp.gamma': (0, 1)}

def objective_bo(**params):
    value = evaluate(params)
    print(f"Params: {params} Value: {value}")
    return value

optimizer = BayesianOptimization(
    f=objective_bo,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=2,
    n_iter=20,
)

data = np.array([(r["target"], r["params"]["mdp.gamma"]) for r in optimizer.res])
plt.scatter(data[:, 1], data[:, 0])
plt.xlabel("Gamma")
plt.ylabel("Score")
plt.show()


plt.plot(data[:, 1])
plt.xlabel("Iteration")
plt.ylabel("Gamma")
plt.show()



print(f"Best params: {optimizer.max}")