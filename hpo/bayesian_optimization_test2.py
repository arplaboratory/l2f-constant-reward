from bayes_opt import BayesianOptimization
from objective2 import evaluate
import numpy as np
import matplotlib.pyplot as plt
import json
import sklearn.gaussian_process.kernels

pbounds = {
    'mdp.reward.position': (0, 50),
    'mdp.gamma': (0, 1),
}
length_scales = [(v[1] - v[0])/10 for k, v in sorted(pbounds.items(), key=lambda x: x[0])]

trace = []
minimize_metric = False

def objective_bo(**params):
    value = evaluate(params)
    print(f"Params: {params} Value: {value}")
    trace.append((params, value))
    return -value if minimize_metric else value

optimizer = BayesianOptimization(
    f=objective_bo,
    pbounds=pbounds,
    random_state=1,
)
optimizer.set_gp_params(normalize_y=True, kernel=sklearn.gaussian_process.kernels.Matern(length_scale=length_scales), n_restarts_optimizer=20) 

optimizer.maximize(
    init_points=0,
    n_iter=200,
)


with open("hpo_results/bayesian_optimization2.json", "w") as f:
    json.dump(trace, f, indent=4)

print(f"Best params: {optimizer.max}")