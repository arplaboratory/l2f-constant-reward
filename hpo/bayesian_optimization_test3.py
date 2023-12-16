from bayes_opt import BayesianOptimization
from objective3 import evaluate
import numpy as np
import matplotlib.pyplot as plt
import json

pbounds = {
    'mdp.gamma': (0.95, 1),
    'mdp.reward.position': (0, 20)
}

trace = []
minimize_metric = True

def objective_bo(**params):
    value = evaluate(params)
    print(f"Params: {params} Value: {value}")
    trace.append((params, value))
    with open("hpo_results_gamma_pos_transfer/bayesian_optimization3.json", "w") as f:
        json.dump(trace, f, indent=4)
    return -value if minimize_metric else value

optimizer = BayesianOptimization(
    f=objective_bo,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=0,
    n_iter=200,
    allow_duplicate_points=True
)



print(f"Best params: {optimizer.max}")