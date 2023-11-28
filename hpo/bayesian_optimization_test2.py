from bayes_opt import BayesianOptimization
from objective2 import evaluate
import numpy as np
import matplotlib.pyplot as plt
import json

pbounds = {
    'mdp.gamma': (0, 1),
    'mdp.reward.position': (0, 50)
}

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
    n_iter=100,
)


with open("hpo_results/bayesian_optimization2.json", "w") as f:
    json.dump(trace, f, indent=4)

print(f"Best params: {optimizer.max}")