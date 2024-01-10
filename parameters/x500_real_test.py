import json, os
from transformations.j_inv import transform as j_inv
from transformations.hovering_throttle import transform as hovering_throttle
name = "x500_real"
experiment = "test"
with open(f'base/{name}_{experiment}.json', 'r') as file:
    base = json.load(file)
with open(f'dynamics/{name}.json', 'r') as file:
    dynamics = json.load(file)
config = {**base, **dynamics}
config['requires_processing'] = 'false'
config['dynamics']['model'] = name
j_inv(config)
hovering_throttle(config)
with open(f'output/{name}_{experiment}.json', 'w') as file:
    json.dump(config, file, indent=4)
