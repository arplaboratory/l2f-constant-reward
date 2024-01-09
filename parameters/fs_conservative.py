import numpy as np
import json, os
name = os.path.splitext(os.path.basename(__file__))[0]
with open(f'base/fs_base.json', 'r') as file:
    base = json.load(file)
with open(f'dynamics/{name}.json', 'r') as file:
    dynamics = json.load(file)
config = {**base, **dynamics}
config['requires_processing'] = 'false'
config['dynamics']['model'] = "fs_base"
if "J" in config["dynamics"]:
    J = np.array(config["dynamics"]["J"])
    config['dynamics']['J_inv'] = np.linalg.inv(J).tolist()
with open(f'output/{name}.json', 'w') as file:
    json.dump(config, file, indent=4)
