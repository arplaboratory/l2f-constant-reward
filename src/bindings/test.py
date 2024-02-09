import l2f
import json
from websockets.sync.client import connect
import time

import numpy as np

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    
    xx2 = 2 * x * x
    yy2 = 2 * y * y
    zz2 = 2 * z * z
    xy2 = 2 * x * y
    xz2 = 2 * x * z
    yz2 = 2 * y * z
    wx2 = 2 * w * x
    wy2 = 2 * w * y
    wz2 = 2 * w * z
    
    R = np.array([[1 - yy2 - zz2, xy2 - wz2, xz2 + wy2],
                  [xy2 + wz2, 1 - xx2 - zz2, yz2 - wx2],
                  [xz2 - wy2, yz2 + wx2, 1 - xx2 - yy2]])
    
    return R



device = l2f.Device()
rng = l2f.RNG()
env = l2f.Environment()
l2f.init(device, env)
state, next_state = [l2f.State() for _ in range(2)]
action = l2f.Action()
observation, next_observation = [l2f.Observation() for _ in range(2)]

l2f.sample_initial_state(device, env, state, rng)

action_value = 0.5
action.motor_command[0] = action_value
action.motor_command[1] = action_value
action.motor_command[2] = action_value
action.motor_command[3] = action_value
l2f.step(device, env, state, action, next_state, rng)
l2f.observe(device, env, state, observation, rng)
l2f.observe(device, env, next_state, next_observation, rng)

print("     observation: ", observation.observation)
print("Next observation: ", next_observation.observation)


model = None
with open('ui_model.json', 'r') as f:
    model = json.load(f)



def add_drone_message(namespace, id):
    add_drone_data = {
        "display_options": {
            "displayActions": True,
            "displayGlobalCoordinateSystem": True,
            "displayIMUCoordinateSystem": True
        },
        "id": id,
        "origin": [
            0,
            0,
            0
        ],
        "model": model
    }
    return {
        "channel": "addDrone",
        "namespace": namespace,
        "data": add_drone_data
    }
def set_state_message(namespace, id, position, orientation):
    R = quaternion_to_rotation_matrix(orientation)
    return {
        "channel": "setDroneState",
        "namespace": namespace,
        "data": {
            "id": id,
            "data": {
                "pose": {
                    "position": position,
                    "orientation": R.tolist()
                },
                "rotor_states": [
                    {"power": 0},
                    {"power": 0},
                    {"power": 0},
                    {"power": 0}
                ]
            }
        }
    }



with connect("ws://localhost:8000/backend") as websocket:
    message = json.loads(websocket.recv())
    namespace = message["data"]["namespace"]
    id = "default"
    print(f"received: {message}")
    websocket.send(json.dumps(add_drone_message(namespace, id)))
    while True:
        l2f.sample_initial_state(device, env, state, rng)
        l2f.observe(device, env, state, observation, rng)
        position = observation.observation[:3]
        orientation = observation.observation[3:3+4]
        websocket.send(json.dumps(set_state_message(namespace, id, position, orientation)))
        time.sleep(1.0)

print("Done")