from bayes_opt import BayesianOptimization
from objective import evaluate
import numpy as np
import matplotlib.pyplot as plt
import json
from params import num_iterations

pbounds = {'mdp.gamma': (0, 1)}

trace = []
metric_name = 'EpisodeLengthMean'
minimize_metric = False

def objective_bo(**params):
    value = evaluate(params)
    print(f"Params: {params} Value: {value}")
    trace.append((params, value))
    return value if not minimize_metric else -value

optimizer = BayesianOptimization(
    f=objective_bo,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=0,
    n_iter=num_iterations,
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

with open("hpo_results/bayesian_optimization.json", "w") as f:
    json.dump(trace, f, indent=4)

print(f"Best params: {optimizer.max}")