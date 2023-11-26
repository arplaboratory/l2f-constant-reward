from ray import train, tune
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hyperopt import HyperOptSearch
from objective import evaluate
import matplotlib.pyplot as plt
import numpy as np
import os

metric_name = 'EpisodeLengthMean'
minimize_metric = False

def objective(config):
    print(f"Cwd: {os.getcwd()}")
    results = evaluate(config)
    results_pp = {k: v if not k.startswith("MaxError") else (v if v > 0 else 1) for k, v in results.items()}
    return {"score": results_pp[metric_name] * (1 if minimize_metric else -1)}


search_space = {
    "mdp.gamma": tune.uniform(0.9, 1.0),
}

# algo = BayesOptSearch(search_space, random_search_steps=4)
algo = HyperOptSearch(search_space, metric="score", mode="min")

tuner = tune.Tuner(objective,
    tune_config=tune.TuneConfig(
        metric="score",
        mode="min",
        search_alg=algo,
        num_samples=100,
    )
)

results = tuner.fit()

data = np.array([(r.metrics["score"], r.config["mdp.gamma"]) for r in results])

plt.scatter(data[:, 1], data[:, 0])
plt.xlabel("Gamma")
plt.ylabel("Score")
plt.show()

print(results.get_best_result(metric="score", mode="min").config)