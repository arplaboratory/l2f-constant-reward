import numpy as np
def transform(config):
    if "J" in config["dynamics"]:
        J = np.array(config["dynamics"]["J"])
        config['dynamics']['J_inv'] = np.linalg.inv(J).tolist()
