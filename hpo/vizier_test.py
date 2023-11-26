from vizier.service import clients
from vizier.service import pyvizier as vz
import vizier
import subprocess
import json
import time
import numpy as np
import random
from .objective import evaluate


print(f"Vizier DB: {vizier.service.VIZIER_DB_PATH}")

def evaluate_vizier(params):
  results = evaluate(params)
  results_pp = {k: v if not k.startswith("MaxError") else (v if v > 0 else 1) for k, v in results.items()}
  return results_pp

study_config = vz.StudyConfig(algorithm='GAUSSIAN_PROCESS_BANDIT')
study_config.search_space.root.add_float_param('mdp.gamma', 0.0, 1.0)
# study_config.search_space.root.add_int_param('x', -2, 2)
# study_config.search_space.root.add_discrete_param('y', [0.3, 7.2])
# study_config.search_space.root.add_categorical_param('z', ['a', 'g', 'k'])
metric_name = 'MaxErrorMean(Position, after 100 steps)'
study_config.metric_information.append(vz.MetricInformation(metric_name, goal=vz.ObjectiveMetricGoal.MINIMIZE))

# Setup client and begin optimization. Vizier Service will be implicitly created.
study = clients.Study.from_study_config(study_config, owner='jonas_eschmann', study_id=str(time.time()))

data = []
# np.array(list({k["mdp.gamma"]:v[metric_name] for k,v in data}.items()))

for i in range(100000):
  suggestions = study.suggest(count=1)
  for suggestion in suggestions:
    params = suggestion.parameters
    objective = evaluate(params)
    data.append((params, objective))
    if np.isinf(objective[metric_name]) or np.isnan(objective[metric_name]):
      suggestion.complete(infeasible_reason="Infeasible")
    else:
      measurement = {}
      measurement[metric_name] = objective[metric_name]
      suggestion.complete(vz.Measurement(measurement))

# {k["mdp.gamma"]:v[metric_name] for k,v in data}
for optimal_trial in study.optimal_trials():
  optimal_trial = optimal_trial.materialize()
  print("Optimal Trial Suggestion and Objective:", optimal_trial.parameters,
        optimal_trial.final_measurement)