import json
import subprocess
import random
import math


with open('../parameters/default.json') as f:
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
def deep_merge(dict1, dict2):
  """
  Recursively merges dict2 into dict1
  """
  for key in dict2:
      if key in dict1:
          if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
              deep_merge(dict1[key], dict2[key])
          else:
              # If dict1[key] is not a dictionary, overwrite it
              dict1[key] = dict2[key]
      else:
          # If key is not in dict1, add it to dict1
          dict1[key] = dict2[key]
  return dict1

def evaluate(params):
  seed = random.randint(0, 1000)
  print(f"Params: {params}")
  merged = deep_merge(default_params, nested_dict(params))
  with open('parameters_temp.json', 'w') as f:
    json.dump(merged, f, indent=4)

  subprocess.call(['../cmake-build-release/src/hpo', "-f", "parameters_temp.json", "-r", "results.json", "-s", str(seed)])

  with open('results.json') as f:
    results = json.loads(f.read())

  episode_length_mean = results['EpisodeLengthMean']
  max_position_error_after_100 = results['MaxErrorMean(Position, after 100 steps)']
  return -max_position_error_after_100 if not math.isinf(max_position_error_after_100) and episode_length_mean > 300 else -1 
#   position_error_after_100_steps_mean = results['MaxErrorMean(Position, after 100 steps)']
#   if episode_length_mean < 400 or position_error_after_100_steps_mean < 0:
#     return 0
#   else:
#     return position_error_after_100_steps_mean