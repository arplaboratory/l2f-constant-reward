import os

original_cwd = os.getcwd()

from ray import train, tune
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hyperopt import HyperOptSearch
import ray
from objective import evaluate
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from params import num_iterations

metric_name = 'EpisodeLengthMean'
minimize_metric = False



def objective(config):
    print(f"Cwd: {os.getcwd()}")
    value = evaluate(config)
    print(f"Params: {config} Value: {value}")
    return {"score": value if minimize_metric else -value}


search_space = {
    "mdp.gamma": tune.uniform(0.0, 1.0),
}

# algo = BayesOptSearch(search_space, random_search_steps=4)
algo = HyperOptSearch(search_space, metric="score", mode="min")

ray.init(local_mode=True)

tuner = tune.Tuner(objective,
    tune_config=tune.TuneConfig(
        metric="score",
        mode="min",
        search_alg=algo,
        num_samples=num_iterations
    ),
)

results = tuner.fit()

data = np.array([(r.metrics["score"], r.config["mdp.gamma"]) for r in results])

plt.scatter(data[:, 1], data[:, 0])
plt.xlabel("Gamma")
plt.ylabel("Score")
plt.show()

trace = [(r.config, r.metrics["score"] * (1 if minimize_metric else -1)) for r in results]

with open(os.path.join(original_cwd, "hpo_results/ray_tune.json"), "w") as f:
    json.dump(trace, f, indent=4)

print(results.get_best_result(metric="score", mode="min").config)