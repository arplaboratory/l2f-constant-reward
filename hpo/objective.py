import json
import subprocess
import random


with open('/home/jonas/phd/projects/rl_for_control/learning_to_fly_hpo/parameters/default.json') as f:
  default_params = json.loads(f.read())

def nested_dict(input_dict):
    def insert(d, keys, value):
        key = keys[0]
        if len(keys) == 1:
            d[key] = value
        else:
            if key not in d:
                d[key] = {}
            insert(d[key], keys[1:], value)

    output_dict = {}
    for key, value in input_dict.items():
        keys = key.split('.')
        insert(output_dict, keys, value)
    return output_dict

def evaluate(params):
  seed = random.randint(0, 1000)
  print(f"Params: {params}")
  merged = {**default_params, **nested_dict(params)}
  with open('parameters_temp.json', 'w') as f:
    json.dump(merged, f, indent=4)

  subprocess.call(['/home/jonas/phd/projects/rl_for_control/learning_to_fly_hpo/cmake-build-release/src/hpo', "-f", "parameters_temp.json", "-r", "results.json", "-s", str(seed)])

  with open('results.json') as f:
    results = json.loads(f.read())

  episode_length_mean = results['EpisodeLengthMean']
  return episode_length_mean
#   position_error_after_100_steps_mean = results['MaxErrorMean(Position, after 100 steps)']
#   if episode_length_mean < 400 or position_error_after_100_steps_mean < 0:
#     return 0
#   else:
#     return position_error_after_100_steps_mean