import optuna
import random
import json
import subprocess
from objective import evaluate

# def objective(trial):
#     x = trial.suggest_float('x', -10, 10)
#     return (x - 2) ** 2
# metric_name = 'MaxErrorMean(Position, after 100 steps)'
metric_name = 'EpisodeLengthMean'
minimize_metric = False


def objective_optuna(trial):
  gamma = trial.suggest_float('mdp.gamma', 0.0, 1.0)
  params = {"mdp": {"gamma": gamma}}
  results = evaluate(params)
  results_pp = {k: v if not k.startswith("MaxError") else (v if v > 0 else 1) for k, v in results.items()}
  return results_pp[metric_name] * (1 if minimize_metric else -1)


study = optuna.create_study()
study.optimize(objective_optuna, n_trials=100)

study.best_params  