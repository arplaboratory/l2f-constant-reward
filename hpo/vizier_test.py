from vizier.service import clients
from vizier.service import pyvizier as vz
import vizier
import subprocess
import json
import time
import numpy as np
import random
from objective import evaluate
from params import num_iterations


print(f"Vizier DB: {vizier.service.VIZIER_DB_PATH}")

trace = []

metric_name = 'EpisodeLengthMean'
minimize_metric = False

trace = []

def evaluate_vizier(params):
  value = evaluate(params)
  trace.append((params, value))
  print(f"Params: {params} Value: {value}")
  return value if minimize_metric else -value

study_config = vz.StudyConfig(algorithm='GAUSSIAN_PROCESS_BANDIT')
study_config.search_space.root.add_float_param('mdp.gamma', 0.0, 1.0)
# study_config.search_space.root.add_int_param('x', -2, 2)
# study_config.search_space.root.add_discrete_param('y', [0.3, 7.2])
# study_config.search_space.root.add_categorical_param('z', ['a', 'g', 'k'])
study_config.metric_information.append(vz.MetricInformation(metric_name, goal=vz.ObjectiveMetricGoal.MINIMIZE))

# Setup client and begin optimization. Vizier Service will be implicitly created.
study = clients.Study.from_study_config(study_config, owner='jonas_eschmann', study_id=str(time.time()))

data = []
# np.array(list({k["mdp.gamma"]:v[metric_name] for k,v in data}.items()))

for i in range(num_iterations):
  suggestions = study.suggest(count=1)
  for suggestion in suggestions:
    params = suggestion.parameters
    objective = evaluate_vizier(params)
    data.append((params, objective))
    if np.isinf(objective) or np.isnan(objective):
      suggestion.complete(infeasible_reason="Infeasible")
    else:
      measurement = {metric_name: objective}
      suggestion.complete(vz.Measurement(measurement))

# {k["mdp.gamma"]:v[metric_name] for k,v in data}
for optimal_trial in study.optimal_trials():
  optimal_trial = optimal_trial.materialize()
  print("Optimal Trial Suggestion and Objective:", optimal_trial.parameters,
        optimal_trial.final_measurement)

with open("hpo_results/vizier.json", "w") as f:
    json.dump(trace, f, indent=4)