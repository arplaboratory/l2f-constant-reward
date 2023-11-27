set -e
. /data/venvs/ray/bin/activate

python3 bayesian_optimization_test.py
python3 optuna_test.py
python3 ray_tune_test.py
python3 vizier_test.py