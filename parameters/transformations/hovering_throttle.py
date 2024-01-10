import numpy as np
def transform(config):
    assert(config["dynamics"]["rotor_thrust_coefficients"][1] == 0, "linear thrust coefficient not handled yet")
    hovering_thrust = config["dynamics"]["mass"] * np.linalg.norm(config["dynamics"]["gravity"]);
    config["dynamics"]["hovering_throttle"] = np.sqrt((hovering_thrust / 4 - config["dynamics"]["rotor_thrust_coefficients"][0]) / config["dynamics"]["rotor_thrust_coefficients"][2]);
