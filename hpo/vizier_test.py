from vizier.service import clients
from vizier.service import pyvizier as vz
import subprocess

def evaluate():
  subprocess.call([''])


study_config = vz.StudyConfig(algorithm='GAUSSIAN_PROCESS_BANDIT')
study_config.search_space.root.add_float_param('gamma', 0.0, 1.0)
# study_config.search_space.root.add_int_param('x', -2, 2)
# study_config.search_space.root.add_discrete_param('y', [0.3, 7.2])
# study_config.search_space.root.add_categorical_param('z', ['a', 'g', 'k'])
study_config.metric_information.append(vz.MetricInformation('MaxErrorMean(Position, after 100 steps)', goal=vz.ObjectiveMetricGoal.MAXIMIZE))

# Setup client and begin optimization. Vizier Service will be implicitly created.
study = clients.Study.from_study_config(study_config, owner='my_name', study_id='example')
for i in range(10):
  suggestions = study.suggest(count=1)
  for suggestion in suggestions:
    params = suggestion.parameters
    objective = evaluate(params['w'], params['x'], params['y'], params['z'])
    suggestion.complete(vz.Measurement({'metric_name': objective}))

for optimal_trial in study.optimal_trials():
  optimal_trial = optimal_trial.materialize()
  print("Optimal Trial Suggestion and Objective:", optimal_trial.parameters,
        optimal_trial.final_measurement)