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
def set_state_message(namespace, id, position, orientation, action=None):
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
                ] if action is None else [
                    {"power": action[0]},
                    {"power": action[1]},
                    {"power": action[2]},
                    {"power": action[3]}
                ]
            }
        }
    }





import torch
class Actor:
    checkpoint = torch.load("best_000032303_33078272_reward_3.374.pth", map_location=torch.device('cpu'))

    model = checkpoint["model"]
    model.keys()

    W_in = model["actor_encoder.self_encoder.0.weight"]
    b_in = model["actor_encoder.self_encoder.0.bias"]
    act_in = torch.nn.Tanh()

    W_1 = model["actor_encoder.self_encoder.2.weight"]
    b_1 = model["actor_encoder.self_encoder.2.bias"]
    act_1 = torch.nn.Tanh()

    W_2 = model["actor_encoder.feed_forward.0.weight"]
    b_2 = model["actor_encoder.feed_forward.0.bias"]
    act_2 = torch.nn.Tanh()

    W_out = model["action_parameterization.distribution_linear.weight"]
    b_out = model["action_parameterization.distribution_linear.bias"]
    act_out = torch.nn.Tanh()

    layers = [
        {"W": W_in, "b": b_in, "act": act_in},
        {"W": W_1, "b": b_1, "act": act_1},
        {"W": W_2, "b": b_2, "act": act_2},
        {"W": W_out, "b": b_out, "act": act_out}
    ]

    def forward(self, x):
        for layer in self.layers:
            x = layer["act"](x @ layer["W"].T + layer["b"])
        return x

actor = Actor()

def thrust2rpm(thrust):
    return (((thrust + 1)/2) ** 0.5) * 2 - 1

with connect("ws://localhost:8000/backend") as websocket:
    message = json.loads(websocket.recv())
    namespace = message["data"]["namespace"]
    id = "default"
    print(f"received: {message}")
    websocket.send(json.dumps(add_drone_message(namespace, id)))
    while True:
        # l2f.sample_initial_state(device, env, state, rng)
        l2f.initial_state(device, env, state)
        for step_i in range(100):
            l2f.observe(device, env, state, observation, rng)
            position_real = state.position
            orientation_real = state.orientation
            position = observation.observation[:3]
            orientation_quaternion = observation.observation[3:3+4]
            orientation_matrix = quaternion_to_rotation_matrix(orientation_quaternion)
            linear_velocity = observation.observation[3+4:3+4+3]
            angular_velocity = observation.observation[3+4+3:3+4+3+3]
            flat_observation = torch.Tensor(np.array([*position, *linear_velocity, *orientation_matrix.ravel(), *angular_velocity]))
            flat_action = actor.forward(flat_observation).detach().numpy()
            print(f"flat_action: {flat_action}")
            action.motor_command = [thrust2rpm(t) for t in flat_action]
            websocket.send(json.dumps(set_state_message(namespace, id, position_real, orientation_real, action=action.motor_command)))
            dt = l2f.step(device, env, state, action, next_state, rng)
            state = next_state
            next_state = l2f.State()

            time.sleep(dt * 10)

print("Done")