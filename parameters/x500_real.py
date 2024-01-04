import numpy as np
import json


with open('base.json', 'r') as file:
    base = json.load(file)
with open('dynamics/x500_real.json', 'r') as file:
    dynamics = json.load(file)
config = {**base, **dynamics}
config['requires_processing'] = 'false'
J = np.array(config["dynamics"]["J"])
config['dynamics']['J_inv'] = np.linalg.inv(J).tolist()
with open('output/default.json', 'w') as file:
    json.dump(config, file, indent=4)
