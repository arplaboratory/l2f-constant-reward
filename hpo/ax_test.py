from ax.service.ax_client import AxClient
from objective import evaluate
import json

metric_name = 'EpisodeLengthMean'
minimize_metric = False

trace = []

def objective_ax(params):
  value = evaluate(params)
  trace.append((params, value))
  return value if minimize_metric else -value

ax_client = AxClient()
ax_client.create_experiment(
    name="learning_to_fly_gamma_experiment",
    parameters=[
        {
            "name": "mdp.gamma",
            "type": "range",
            "bounds": [.0, 1.0],
            "value_type": "float",
        }
    ],
    objective_name="learning_to_fly_gamma",
    minimize=True,
)

for _ in range(20):
    parameters, trial_index = ax_client.get_next_trial()
    ax_client.complete_trial(trial_index=trial_index, raw_data=objective_ax(parameters))

with open("hpo_results/ax.json", "w") as f:
    json.dump(trace, f, indent=4)

best_parameters, metrics = ax_client.get_best_parameters()