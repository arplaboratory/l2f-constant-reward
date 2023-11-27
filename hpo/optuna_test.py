import optuna
import random
import json
import subprocess
from objective import evaluate
from params import num_iterations

# def objective(trial):
#     x = trial.suggest_float('x', -10, 10)
#     return (x - 2) ** 2
# metric_name = 'MaxErrorMean(Position, after 100 steps)'
metric_name = 'EpisodeLengthMean'
minimize_metric = False

trace = []

def objective_optuna(trial):
  gamma = trial.suggest_float('mdp.gamma', 0.0, 1.0)
  params = {"mdp.gamma": gamma}
  value = evaluate(params)
  trace.append((params, value))
  return value if minimize_metric else -value


study = optuna.create_study()
study.optimize(objective_optuna, n_trials=num_iterations)

with open("hpo_results/optuna.json", "w") as f:
    json.dump(trace, f, indent=4)

study.best_params  