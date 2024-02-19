import numpy as np
def transform(config):
    if "rotor_thrust_coefficients" in config["dynamics"]:
        assert(config["dynamics"]["rotor_thrust_coefficients"][1] == 0, "linear thrust coefficient not handled yet")
        hovering_thrust = config["dynamics"]["mass"] * np.linalg.norm(config["dynamics"]["gravity"]);
        config["dynamics"]["hovering_throttle"] = np.sqrt((hovering_thrust / 4 - config["dynamics"]["rotor_thrust_coefficients"][0]) / config["dynamics"]["rotor_thrust_coefficients"][2]); # absolute (RPM space)
        config["dynamics"]["hovering_throttle_relative"] = (config["dynamics"]["hovering_throttle"] - config["dynamics"]["action_limit"]["lower_bound"]) / (config["dynamics"]["action_limit"]["upper_bound"] - config["dynamics"]["action_limit"]["lower_bound"]); # relative to the action space
